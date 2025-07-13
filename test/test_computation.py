from model_backend import computation_graph as cg
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import json


def test_computation_graph_invalid_config(computation_graph_invalid_highlevel):
    # test with an invalid config file.
    with pytest.raises(ValueError, match="Configuration verification failed:"):
        cg.ComputationGraph(computation_graph_invalid_highlevel)


def test_computation_graph_invalid_modules(computation_graph_invalid_modules):
    assert 3 == 7


def test_computation_graph_invalid_functions(computation_graph_invalid_func):
    assert 3 == 7


def test_computation_graph_invalid_graph(computation_graph_multiple_sink_nodes):
    assert 3 == 7


def test_computation_graph_working_initialization(computation_graph_config):
    assert 3 == 7


def test_computation_graph_properties(computation_graph_config):
    assert 3 == 7


@pytest.mark.parameterize(
    "scheduler", ["synchronous", "multithreaded", "multiprocessing", "distributed"]
)
def test_computation_graph_execution_schedulers(scheduler):
    assert 3 == 7
