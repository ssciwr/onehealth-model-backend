"""
This module defines the Computation class. This class represents a model as a computation directed acyclic graph (DAG) that executes a series of tasks which together represent the run of a given oneHealth model and manages the setup and execution of such a graph.
"""

# dask needed for parallel execution and lazy evaluation
import dask

# stdlib imports
import json
from typing import Callable, Any
from pathlib import Path
import importlib
import inspect

# internals
from . import JModel
from . import utils


class ModelComputation:
    """A class to represent a computation DAG that executes a series of tasks which together represent the run of a given oneHealth model. These models are defined as combinations of functions known to the class.
    Modules are a loose collection of functions that are registered with the class and combined into a computational graph to create a functional system. Therefore, functions are registered as either part of a module or as utility functions, e.g., if they are used by multiple modules. The computational graph is built from these functions and executed in via dask tasks to allow for parallel, lazy execution and efficient resource management. Computations can be combined freely from the functions registered with different modules.

    Attributes:
        modules (dict[str, Any]): A dictionary of modules, where each module is a module object imported from a given path.
        module_functions (dict[str, dict[str, Callable]]): A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
        dask_graph (dict[str, dask.taskspec.Task | dask.taskspec.DataNode]): A dictionary representing the Dask computational graph, where each node is a task or data node.
        config (dict[str, Any]): A configuration dictionary for the computation, containing paths for input and output data, as well as the computational graph structure.
    """

    modules: dict[str, Any] = {}
    module_functions: dict[str, dict[str, Callable]] = {}
    config: dict[str, Any] = None  # Configuration for the computation
    sink_node: dask.delayed.Delayed | None = (
        None  # The sink node of the computational graph
    )

    def __init__(self, config: dict[str, Any]):
        """_summary_

        Args:
            config (dict[str, Any]): _description_

        Raises:
            ValueError: _description_
        """
        config_valid, msg = self._verify(config)

        if not config_valid:
            raise ValueError(f"Configuration verification failed: {msg}")

        self.config = config

        # load needed code
        self.modules = self._load_modules(config)
        self.module_functions = self._get_functions_from_module(self.modules)

        # build the computational graph and find the sink node which we use to execute the graph
        self.sink_node = self._build_dag(config)

        # set the dask scheduler
        dask.config.set(scheduler=config["execution"]["scheduler"])

    def _load_modules(self, config: dict[str, Any]) -> dict[str, Any]:
        """_summary_

        Args:
            config (dict[str, Any]): _description_

        Raises:
            ImportError: _description_

        Returns:
            dict[str, Any]: _description_
        """
        modules = {"JModel": JModel, "utilities": utils}

        for module_name, module_info in config["modules"].items():
            module_path = module_info["path"]
            try:
                mod = importlib.import_module(module_path)
                modules[module_name] = mod
            except ImportError as e:
                raise ImportError(f"Could not import module {module_name}: {e}")
        return modules

    def _get_functions_from_module(
        self, modules: dict[str, Any]
    ) -> dict[str, dict[str, Callable]]:
        """_summary_

        Args:
            modules (dict[str, Any]): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            dict[str, dict[str, Callable]]: _description_
        """
        module_functions = {}

        for module_name, module in modules.items():
            if not hasattr(module, "__dict__"):
                raise ValueError(
                    f"Module {module_name} does not have a __dict__ attribute."
                )

            try:
                functions = {
                    name: func
                    for name, func in inspect.getmembers(module, inspect.isfunction)
                }
                module_functions[module_name] = functions
            except Exception as e:
                raise ValueError(
                    f"Could not retrieve functions from module {module_name}: {e}"
                )
        return module_functions

    def _find_sink_node(self, config: dict[str, Any]) -> str:
        """_summary_

        Args:
            config (dict[str, Any]): _description_

        Raises:
            ValueError: _description_

        Returns:
            str: _description_
        """
        all_inputs = []

        for node in config["graph"].values():
            if "input" in node and isinstance(node["input"], list):
                all_inputs.extend(node["input"])
        all_inputs = set(all_inputs)

        sink_node = None
        for node_name, _ in config["graph"].items():
            if node_name not in all_inputs:
                if sink_node is not None:
                    raise ValueError(
                        "Multiple sink nodes found in the computational graph."
                    )
                sink_node = node_name

        if sink_node is None:
            raise ValueError("No sink node found in the computational graph.")
        return sink_node

    def _build_dag(self, config: dict[str, Any]) -> dask.delayed.Delayed:
        """_summary_

        Args:
            config (dict[str, Any]): _description_

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            dict[str, daskTS.Task | daskTS.DataNode]: _description_
        """
        tmp_delayed = {}

        sink_node_name = self._find_sink_node(config)

        # forward pass through the dict -> build up all the dask.delayed objects
        for current_node in config["graph"].keys():
            task_func_name = config["graph"][current_node]["function"]
            task_func_module = config["graph"][current_node]["module"]

            if task_func_module not in self.module_functions:
                raise ValueError(
                    f"Module {task_func_module} not found in registered modules."
                )

            if task_func_name not in self.module_functions[task_func_module]:
                raise ValueError(
                    f"Function {task_func_name} not found in module {task_func_module}."
                )

            task_func = self.module_functions[task_func_module][task_func_name]

            task = dask.delayed(task_func)
            tmp_delayed[current_node] = task

        delayed_tasks = {}
        for current_node, node_info in config["graph"].items():
            task = tmp_delayed[current_node]
            input_nodes = node_info["input"]
            args = node_info["args"] if node_info["args"] is not None else []
            kwargs = node_info["kwargs"] if node_info["kwargs"] is not None else {}

            # resolve input nodes
            resolved_inputs = [tmp_delayed[input_node] for input_node in input_nodes]

            # create the task with the resolved inputs
            if len(resolved_inputs) > 0:
                task = task(*resolved_inputs, *args, **kwargs)
            else:
                task = task(*args, **kwargs)
            delayed_tasks[current_node] = task

        return delayed_tasks[sink_node_name]

    def _verify(self, config: dict[str, Any]) -> bool:
        """_summary_

        Args:
            config (dict[str, Any]): _description_

        Returns:
            bool: _description_
        """
        needed_high_level_keys = ["data", "graph", "execution"]
        if not all(key in config for key in needed_high_level_keys):
            return (
                False,
                f"Configuration is missing required keys. Required keys are {needed_high_level_keys}.",
            )

        if not all(key in config["data"] for key in ["input", "output"]):
            return (
                False,
                "Data configuration is missing required keys. Required keys are 'input' and 'output'.",
            )

        if "scheduler" not in config["execution"]:
            return False, "Execution configuration is missing 'scheduler' key."

        if config["execution"]["scheduler"] not in [
            "synchronous",
            "multithreaded",
            "multiprocessing",
            "distributed",
        ]:
            return (
                False,
                f"Unsupported scheduler: {config['execution']['scheduler']}. Supported schedulers are 'synchronous', 'multithreaded', 'multiprocessing'.",
            )

        # verify the computation structure
        # all nodes in the computational graph must define their input nodes and the function they execute, as well as additional parameters they might need. They also need to define the name of the module they are part of.
        for node, value in config["graph"].items():
            if value is None or not isinstance(value, dict):
                return False, f"Node {node} is not a dict."

            if any(
                key not in value
                for key in ["function", "input", "args", "kwargs", "module"]
            ):
                return (
                    False,
                    f"Node {node} is missing required keys. Required keys are 'function', 'input', 'args', 'kwargs', and 'module'.",
                )

            if "path" not in value["module"] and value["module"]["name"] not in [
                "JModel",
                "utilities",
            ]:
                return (
                    False,
                    f"Module {value['module']['name']} does not have a path defined and is not a known default module.",
                )

            for input_node in value["input"]:
                if input_node not in config["graph"]:
                    return (
                        False,
                        f"input node {input_node} of node {node} not found in graph",
                    )

            if not isinstance(value["input"], list):
                return False, f"input nodes for node {node} must be a list"
            # the parameters must either be a dictionary of parameters or None
            if not isinstance(value["args"], list) and value["args"] is not None:
                return False, f"arguments for node {node} must be a list"

            if not isinstance(value["kwargs"], dict) and value["kwargs"] is not None:
                return False, f"keyword arguments for node {node} must be a dict"

        return True, "Configuration is valid."

    def execute(self, client: dask.distributed.Client = None):
        if self.sink_node is None:
            raise ValueError("Sink node is not defined. Cannot execute the graph.")

        if self.scheduler is None:
            raise ValueError("Scheduler is not defined. Cannot execute the graph.")

        return self.sink_node.compute(scheduler=self.scheduler, client=client)

    def visualize(self):
        """Visualizes the computational graph.

        Raises:
            ValueError: If the sink node is not defined.

        Returns:
            Any: The visualization of the sink node.
        """
        if self.sink_node is None:
            raise ValueError("Sink node is not defined. Cannot visualize the graph.")
        return self.sink_node.visualize()

    @property
    def known_modules(self) -> dict[str, Any]:
        """Returns the known modules dictionary."""
        return list(self.modules.keys())

    @property
    def known_functions(self) -> dict[str, list[str]]:
        """Returns the known functions dictionary."""
        return {
            module_name: list(functions.keys())
            for module_name, functions in self.module_functions.items()
        }

    @classmethod
    def from_config(cls, path_to_config: str | Path) -> "ModelComputation":
        """Creates a `ModelComputation` instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
