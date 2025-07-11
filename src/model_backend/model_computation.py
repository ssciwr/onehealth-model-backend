"""
This module defines the Computation class. This class represents a model as a computation directed acyclic graph (DAG) that executes a series of tasks which together represent the run of a given oneHealth model and manages the setup and execution of such a graph.
"""

import dask.task_spec as daskTS
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
    dask_graph: dict[str, daskTS.Task | daskTS.DataNode] = (
        None  # Placeholder for the Dask graph
    )
    config: dict[str, Any] = None  # Configuration for the computation

    def __init__(self, config: dict[str, Any]):
        config_valid, msg = self._verify(config)
        if not config_valid:
            raise ValueError(f"Configuration verification failed: {msg}")
        self.config = config
        self.modules = self._load_modules(config)
        self.module_functions = self._get_functions_from_module(self.modules)
        self.computational_graph = self._build_comp_graph(config)

    def _load_modules(self, config: dict[str, Any]) -> dict[str, Any]:
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

    def _find_computations_node(self, config: dict[str, Any]) -> str:
        all_inputs = []

        for node in config["graph"].values():
            if "input" in node and isinstance(node["input"], list):
                all_inputs.extend(node["input"])
        all_inputs = set(all_inputs)

        sink = None
        for node_name, _ in config["graph"].items():
            if node_name not in all_inputs:
                if sink is not None:
                    raise ValueError(
                        "Multiple sink nodes found in the computational graph."
                    )
                sink = node_name
        return sink

    def _build_comp_graph(
        self, config: dict[str, Any]
    ) -> dict[str, daskTS.Task | daskTS.DataNode]:
        comp_graph: dict[str, daskTS.Task | daskTS.DataNode] = {}
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
            task_params = (
                config["graph"][current_node]["params"].values()
                if config["graph"][current_node]["params"] is not None
                else []
            )
            input_funcs = config["graph"][current_node]["inputs"]

            dependencies = [daskTS.TaskRef(node) for node in input_funcs]
            comp_graph[current_node] = daskTS.Task(
                current_node, task_func, [*dependencies, *task_params]
            )

        return comp_graph

    def _verify(self, config: dict[str, Any]) -> bool:
        needed_high_level_keys = ["data", "graph", "execution"]
        if not all(key in config for key in needed_high_level_keys):
            return False

        if not all(key in config["data"] for key in ["input", "output"]):
            return False

        if "scheduler" not in config["execution"]:
            return False

        # all nodes in the computational graph must define their input nodes and the function they execute, as well as additional parameters they might need. They also need to define the name of the module they are part of.
        for node, value in config["graph"].items():
            if value is None or not isinstance(value, dict):
                return False, f"Node {node} is not a dict."

            if any(
                key not in value for key in ["function", "input", "params", "module"]
            ):
                return (
                    False,
                    f"Node {node} is missing required keys. Required keys are 'function', 'input', 'params', and 'module'.",
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
            if not isinstance(value["params"], dict) and value["params"] is not None:
                return False, f"parameters for node {node} must be a dictionary"

        return True, "Configuration is valid."

    def execute(self):
        pass

    def visualize(self):
        pass

    @property
    def dask_computational_graph(self) -> Any:
        """Returns the Dask graph for the computation."""
        if self.dask_graph is None:
            raise ValueError("Dask graph has not been built yet.")
        return self.dask_graph

    @property
    def raw_computational_graph(self) -> Any:
        """Returns the computational graph for the computation."""
        if self.computational_graph is None:
            raise ValueError("Computational graph has not been built yet.")
        return self.computational_graph

    @property
    def known_modules(self) -> dict[str, Any]:
        """Returns the known modules dictionary."""
        return self.modules

    @property
    def known_utilities(self) -> dict[str, Callable]:
        """Returns the known utilities dictionary."""
        return self.utilities

    @classmethod
    def from_config(cls, path_to_config: str | Path) -> "ModelComputation":
        """Creates a `ModelComputation` instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
