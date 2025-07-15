import pytest
import numpy as np
import xarray as xr
from pathlib import Path


def make_rioxarray_testdata(
    path: Path, valid: bool = True, resolution: int = 10
) -> xr.Dataset:
    if valid:
        lon = np.linspace(-180, 180, resolution)
        lat = np.linspace(-90, 90, resolution)
    else:
        lon = np.linspace(-120, 120, resolution)
        lat = np.linspace(-70, 76, resolution)
    data = np.random.rand(resolution, resolution) * 10

    # Create dataset with proper dimension order for geospatial data
    ds = xr.Dataset(
        data_vars={
            "t2m": (("latitude", "longitude"), data),
        },
        coords={
            "latitude": lat,
            "longitude": lon,
        },
    )

    if (path / "test_data.nc").exists():
        # Remove existing file if it exists
        (path / "test_data.nc").unlink()

    # Save to netCDF
    ds.to_netcdf(path / "test_data.nc", mode="w", format="NETCDF4")

    # First load with netCDF engine
    with xr.open_dataset(path / "test_data.nc") as data:
        ds = data.compute()
    return ds


@pytest.fixture
def make_test_data(tmp_path) -> xr.Dataset:
    return make_rioxarray_testdata(tmp_path, valid=True, resolution=50)


@pytest.fixture
def make_invalid_test_data(tmp_path) -> xr.Dataset:
    return make_rioxarray_testdata(tmp_path, valid=False, resolution=50)


# for execution pipeline
@pytest.fixture
def computation_graph_config(tmp_path):
    return {
        "load_data": {
            "function": "load_data",
            "module": "./test/computation_module.py",
            "args": [
                "./test/computation_test_data.csv",
            ],
            "kwargs": {},
            "input": [],
        },
        "add": {
            "module": "./test/computation_module.py",
            "function": "add",
            "input": [
                "load_data",
            ],
            "args": [
                2,
            ],
            "kwargs": {},
        },
        "multiply": {
            "module": "./test/computation_module.py",
            "function": "multiply",
            "input": ["load_data", "add"],
            "args": [],
            "kwargs": {},
        },
        "subtract": {
            "module": "./test/computation_module.py",
            "function": "subtract",
            "input": ["add", "multiply"],
            "args": [],
            "kwargs": {},
        },
        "affine": {
            "module": "./test/computation_module.py",
            "function": "affine",
            "input": [
                "subtract",
            ],
            "args": [],
            "kwargs": {"b": 5, "a": 2},
        },
        "save": {
            "function": "save_data",
            "module": "./test/computation_module.py",
            "input": ["affine"],
            "args": [str(tmp_path / "output.csv")],
            "kwargs": {},
        },
    }


@pytest.fixture
def computation_graph_working(computation_graph_config):
    return {
        "graph": computation_graph_config,
        "execution": {
            "scheduler": "synchronous",
        },
    }


@pytest.fixture
def computation_graph_invalid_highlevel(computation_graph_config):
    return {
        "graph": computation_graph_config,
        "invalid": {
            "foo": "bar",
        },
    }


def computation_graph_invalid_execution(computation_graph_config):
    return {
        "graph": computation_graph_config,
        "execution": {
            "scheduler": "invalid_scheduler",
        },
    }


@pytest.fixture
def computation_graph_multiple_sink_nodes():
    return {
        "graph": {
            "load_data": {
                "function": "load_data",
                "module": "./test/computation_module.py",
                "input": [],
                "args": ["./data.csv"],
                "kwargs": {},
            },
            "add": {
                "module": "./test/computation_module.py",
                "function": "add",
                "input": ["load_data"],
                "args": [],
                "kwargs": {},
            },
            "multiply": {
                "module": "./test/computation_module.py",
                "function": "multiply",
                "input": ["load_data"],
                "args": [],
                "kwargs": {},
            },
        },
        "execution": {
            "scheduler": "synchronous",
        },
    }


@pytest.fixture
def computation_graph_invalid_modules():
    return {
        "execution": {
            "scheduler": "synchronous",
        },
        "graph": {
            "load_data": {
                "function": "load_data",
                "module": "./test/computation_module.py",
                "input": [],
                "args": ["./data.csv"],
                "kwargs": {},
            },
            "invalid_module": {
                "module": "./non_existent_module",
                "function": "some_function",
                "input": ["load_data"],
                "args": [],
                "kwargs": {},
            },
            "save": {
                "function": "save_data",
                "module": "./test/computation_module.py",
                "input": ["invalid_module"],
                "args": ["./output.csv"],
                "kwargs": {},
            },
        },
    }


@pytest.fixture
def computation_graph_invalid_func():
    return {
        "execution": {
            "scheduler": "synchronous",
        },
        "graph": {
            "load_data": {
                "function": "load_data",
                "module": "./test/computation_module.py",
                "args": ["./data.csv"],
                "kwargs": {},
                "input": [],
            },
            "invalid_function": {
                "module": "./test/computation_module.py",
                "function": "non_existent_function",
                "input": ["load_data"],
                "args": [],
                "kwargs": {},
            },
            "save": {
                "function": "save_data",
                "module": "./test/computation_module.py",
                "input": ["invalid_function"],
                "args": ["./output.csv"],
                "kwargs": {},
            },
        },
    }
