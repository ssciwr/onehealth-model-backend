""" 
This module defines the Computation class. This class represents a model as a computation directed acyclic graph (DAG) that executes a series of tasks which together represent the run of a given oneHealth model and manages the setup and execution of such a graph.
"""
import dask
import json
from typing import Callable, Any
from pathlib import Path


class Computation:
    """A class to represent a computation DAG that executes a series of tasks which together represent the run of a given oneHealth model.
    modules are defined as a loose collection of functions that are registered with the class and combined into a computational graph to create a functional system. Functions are registered as either part of a model or as utility functions, e.g., if they are used by multiple modules. The computational graph is built from these functions and executed in a Dask environment to allow for parallel, lazy execution and efficient resource management. Computations can be combined freely from the functions registered with different modules.
    """

    modules: dict[str, Any] = {}
    utilities: dict[str, Callable] = {}
    computational_graph: Any = None  # Placeholder for the computational graph
    dask_graph: Any = None  # Placeholder for the Dask graph
    config: dict[str, Any] = None  # Configuration for the computation

    def __init__(self, config: dict[str, Any]):
        if not self._verify(config):
            raise ValueError("Configuration verification failed.")
        self.config = config

        # TODO: find all the functions that we need to register from which are either provided or are available in the modules repository

        # TODO: build the computational graph

        # TODO: visualize it

        # TODO: save the visualized graph and the configuraion to the output path

    def _build_pipeline(self, config: dict[str, Any]) -> Callable:
        pass

    def _create_output(self):
        pass

    def _verify(self, config: dict[str, Any]) -> bool:
        pass 

    def _add_new_model_full(self, config: dict[str, Any]) -> None:
        pass

    def register_new_utility(self, func: Callable) -> None:
        pass

    def register_new_model(self, name: str) -> None:
        pass

    def register_new_model_propagator(self, model: str, func: Callable) -> None:
        pass

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
    def from_config(cls, path_to_config: str | Path) -> "Computation":
        """Creates a Pipeline instance from a configuration dictionary read from a json file."""
        with open(path_to_config, "r") as f:
            config = json.load(f)
        return cls(config)
