"""
This module defines the ComputationGraph class. This class represents a model as a directed acyclic graph (DAG) that executes a series of interdependent tasks which together represent the run of a given oneHealth model and manages the setup and execution of such a graph.
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


class ComputationGraph:
    """A class to represent a computation DAG that executes a series of tasks which together represent the run of a given oneHealth model. These models are defined as combinations of functions known to the class.
    Modules are a loose collection of functions that are registered with the class and combined into a computational graph to create a functional system. Therefore, functions are registered as either part of a module or as utility functions, e.g., if they are used by multiple modules. The computational graph is built from these functions and executed in via dask tasks to allow for parallel, lazy execution and efficient resource management. Computations can be combined freely from the functions registered with different modules.

    Attributes:
        modules (dict[str, Any]): A dictionary of modules, where each module is a module object imported from a given path.
        module_functions (dict[str, dict[str, Callable]]): A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
        task_graph (dict[str, dask.delayed.Delayed]): A dictionary representing the Dask computational graph, where each node is a dask.delayed object.
        config (dict[str, Any]): A configuration dictionary for the computation, the computational graph structure.
        sink_node (dask.delayed.Delayed | None): The sink node of the computational graph, which is the final node that triggers the execution of the entire computation.
    """

    modules: dict[str, Any] = {}
    module_functions: dict[str, dict[str, Callable]] = {}
    config: dict[str, Any] = None  # Configuration for the computation
    sink_node: dask.delayed.Delayed | None = (
        None  # The sink node of the computational graph
    )
    task_graph: dict[str, dask.delayed.Delayed] | None = None

    def __init__(self, config: dict[str, Any]):
        """Initialize the computation graph from the given configuration.
        This method verifies the configuration, loads the necessary modules, retrieves the functions from the modules, builds the computational graph, and sets the Dask scheduler.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ValueError: If the configuration is invalid.
        """
        config_valid, msg = self._verify_config(config)

        if not config_valid:
            raise ValueError(f"Configuration verification failed: {msg}")

        self.config = config

        # load needed code
        self.modules = self._load_modules(config)
        self.module_functions = self._get_functions_from_module(self.modules)

        # build the computational graph and find the sink node which we use to execute the graph
        self.task_graph, self.sink_node = self._build_dag(config)

        # set the dask scheduler
        dask.config.set(scheduler=config["execution"]["scheduler"])

    def _load_modules(self, config: dict[str, Any]) -> dict[str, Any]:
        """Load modules specified in the configuration. It additionally imports the internal default modules (JModel, utilities), which are always available.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Raises:
            ImportError: If a module cannot be imported.

        Returns:
            dict[str, Any]: A dictionary mapping module names to module objects.
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
        """Find all functions in the given modules and return a dictionary mapping module names to dictionaries of function names and their corresponding callable objects.

        Args:
            modules (dict[str, Any]): A dictionary of module names to module objects.

        Raises:
            ValueError: If a module does not have a __dict__ attribute.
            ValueError: If functions cannot be retrieved from a module.

        Returns:
            dict[str, dict[str, Callable]]: A dictionary mapping module names to dictionaries of function names and their corresponding callable objects.
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
                    if name[0] != "_"  # Exclude private functions
                }
                module_functions[module_name] = functions
            except Exception as e:
                raise ValueError(
                    f"Could not retrieve functions from module {module_name}: {e}"
                )
        return module_functions

    def _find_sink_node(self, config: dict[str, Any]) -> str:
        """Find the sink node in the computational graph.

        Args:
            config (dict[str, Any]): Configuration dictionary containing the computational graph structure.

        Raises:
            ValueError: If multiple sink nodes are found.
            ValueError: If no sink node is found.

        Returns:
            str: The name of the sink node.
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

    def _build_dag(
        self, config: dict[str, Any]
    ) -> tuple[dict[str, dask.delayed.Delayed], dask.delayed.Delayed]:
        """Build the Dask computational graph from the configuration.
            The returned executable graph is a dictionary mapping node names to Dask delayed objects, and the returned sink node is the final node in the graph that triggers the execution of the entire computation. There must only be one sink node in the graph.
            If there are multiple sink nodes, an error is raised.
        Args:
            config (dict[str, Any]): Configuration dictionary containing the computational graph structure as name: {
                "function": function_name,
                "input": [input_node_name1, input_node_name2, ...],
                "args": [args],
                "kwargs": {kwargs},
                "module": module_name
            }

        Raises:
            ValueError: A module is not found in the registered modules.
            ValueError:  A function is not found in the module.

        Returns:
            tuple[dict[str, dask.delayed.Delayed], dask.delayed.Delayed]: A tuple containing the Dask computational graph and the sink node.

        """
        # TODO: function isn't particularly elegant or tasteful. Streamline and make simpler.

        # building the graph consists of three steps:
        # 1. find the name of the sink node. There must only be one or we cannot guarantee that the graph will be executed completely.
        # 2. create a temporary dict that holds the dask.delayed objects for each computation node
        # 3. create the actual dask.delayed tasks with their inputs and parameters and respect their interdependencies.

        # 1) find sink node
        sink_node_name = self._find_sink_node(config)

        # 2) create a temporary dict that holds the dask.delayed objects for each computation node
        tmp_delayed = {}

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

        # 3) create the delayed tasks with their inputs and parameters and respect their interdependencies. This builds up the actual computational
        # graph. We later use the sink node to trigger the execution of the
        # entire graph. The dask graph is saved as a class attribute for later
        # use, which might come in handy for debugging, visualization or
        # development
        delayed_tasks = {}
        for current_node, node_info in config["graph"].items():
            # get all the stuff the task needs first
            task = tmp_delayed[current_node]
            input_nodes = node_info["input"]
            args = node_info["args"] if node_info["args"] is not None else []
            kwargs = node_info["kwargs"] if node_info["kwargs"] is not None else {}

            # resolve input nodes using the tmp_delayed dict that was build earlier
            resolved_inputs = [tmp_delayed[input_node] for input_node in input_nodes]

            # create the task with the resolved inputs
            if len(resolved_inputs) > 0:
                task = task(*resolved_inputs, *args, **kwargs)
            else:
                task = task(*args, **kwargs)
            delayed_tasks[current_node] = task

        return delayed_tasks, delayed_tasks[sink_node_name]

    def _verify_config(self, config: dict[str, Any]) -> tuple[bool, str]:
        """Verify the configuration dictionary.

        Args:
            config (dict[str, Any]): The configuration dictionary.

        Returns:
            bool, str: A tuple containing a boolean indicating whether the configuration is valid and an error message if it is not.
        """

        # verify the high-level structure of the configuration
        needed_high_level_keys = ["graph", "execution"]
        if not all(key in config for key in needed_high_level_keys):
            return (
                False,
                f"Configuration is missing required keys. Required keys are {needed_high_level_keys}.",
            )

        # we need to have a dask scheduler defined in the execution section
        if "scheduler" not in config["execution"]:
            return False, "Execution configuration is missing 'scheduler' key."

        # ... and it must be one of the supported schedulers
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

        # verify the computation structure.
        for node, value in config["graph"].items():
            # verify that the node is a dict
            if value is None or not isinstance(value, dict):
                return False, f"Node {node} is not a dict."

            # all nodes that define a computation node must have the name of the function to call, a list of nodes that need to run before this one and that are used as inputs, as well as additional arguments and keyword arguments that are passed to the function. The latter two can be empty or None, but the keys must be persent ot make this choice explicit.
            if any(
                key not in value
                for key in ["function", "input", "args", "kwargs", "module"]
            ):
                return (
                    False,
                    f"Node {node} is missing required keys. Required keys are 'function', 'input', 'args', 'kwargs', and 'module'.",
                )

            # the module specifications must have paths if they are not one of the default modules, sucht that we know where to find the module.
            if "name" not in value["module"]:
                return (
                    False,
                    f"Module {value['module']} does not have a name defined.",
                )

            if "path" not in value["module"] and value["module"]["name"] not in [
                "JModel",
                "utilities",
            ]:
                return (
                    False,
                    f"Module {value['module']['name']} does not have a path defined and is not a known default module.",
                )

            # the input nodes must be explicitly specified and must be present # in the graph somewhere, otherwhise we cannot resolve them.
            for input_node in value["input"]:
                if input_node not in config["graph"]:
                    return (
                        False,
                        f"input node {input_node} of node {node} not found in graph",
                    )

            # the input nodes must be a list of names of other nodes
            if not isinstance(value["input"], list):
                return False, f"input nodes for node {node} must be a list"

            # the positional arguments and keyword arguments must be a list and a dict, respectively, or None
            if not isinstance(value["args"], list) and value["args"] is not None:
                return False, f"arguments for node {node} must be a list"

            if not isinstance(value["kwargs"], dict) and value["kwargs"] is not None:
                return False, f"keyword arguments for node {node} must be a dict"

        return True, "Configuration is valid."

    def execute(self, client: dask.distributed.Client = None):
        """Executes the computational graph.

        Args:
            client (dask.distributed.Client, optional): The Dask client to use for execution if the computation should be executed on a cluster. If None, will use the local machine. Defaults to None. For more on the Dask client, see https://distributed.dask.org/en/stable/client.html.

        Raises:
            ValueError: If the sink node is not defined.
            ValueError: If the scheduler is not defined.

        Returns:
            Any: The result of the computation.
        """
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
            Any: The visualization of the sink node as returned by the dask.delayed.Delayed.visualize method.
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
    def from_config(cls, path_to_config: str | Path) -> "ComputationGraph":
        """Creates a `ComputationGraph` instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
