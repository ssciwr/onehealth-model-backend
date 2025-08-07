from heiplanet_models import Jmodel as jm
from heiplanet_models import computation_graph as cg
import xarray as xr
from pathlib import Path
import pandas as pd
import pytest
import math
import numpy as np
import json


def test_jmodel_initialization():
    # Test valid initialization

    path = Path.cwd() / "test" / "test_r0.csv"

    model = jm.setup_modeldata(
        input="input_data.csv",
        output="output_data.csv",
        r0_path=path,
        run_mode="parallelized",
        grid_data_baseurl="https://example.com/grid_data",
        nuts_level=3,
        resolution="10M",
        year=2024,
    )

    assert model.input == "input_data.csv"
    assert model.output == "output_data.csv"
    assert model.run_mode == "parallelized"
    assert isinstance(model.r0_data, pd.DataFrame)
    assert model.grid_data_baseurl == "https://example.com/grid_data"
    assert model.nuts_level == 3
    assert model.resolution == "10M"
    assert model.year == 2024
    assert math.isclose(model.min_temp, 8.6)
    assert math.isclose(model.max_temp, 13.0)

    with pytest.raises(ValueError):
        jm.setup_modeldata(
            input="input_data.csv",
            output=None,
            r0_path=str(path),
            run_mode="forbidden",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )
    with pytest.raises(ValueError):
        jm.setup_modeldata(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=None,
            run_mode="forbidden",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

    with pytest.raises(ValueError):
        jm.setup_modeldata(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=str(path),
            run_mode="invalid_mode",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

    with pytest.raises(ValueError):
        jm.setup_modeldata(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=path,
            run_mode="parallelized",  # Invalid run mode
            grid_data_baseurl=None,
            nuts_level=3,
            resolution="10M",
            year=2024,
        )


def test_read_default_config():
    # Test reading the default configuration file
    config = jm.read_default_config()
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "graph" in config, "Config should contain 'graph' key"
    assert (
        "setup_modeldata" in config["graph"]
    ), "Graph should contain 'setup_modeldata' key"
    assert (
        "kwargs" in config["graph"]["setup_modeldata"]
    ), "Setup model data should have 'kwargs' key"
    assert (
        "input" in config["graph"]["setup_modeldata"]["kwargs"]
    ), "Input path should be specified in the config"


def test_model_read_input_data(make_test_data, tmp_path):
    with make_test_data as data:
        model = jm.setup_modeldata(
            input=tmp_path / "test_data.nc",
            output="output_data.csv",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="parallelized",
            grid_data_baseurl="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

        read_data = jm.read_input_data(model).compute()

        assert isinstance(read_data, xr.Dataset), "should be xr dataset"
        assert "t2m" in read_data.data_vars, "correct data dim should be in the dataset"
        assert read_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"

        assert (
            read_data.longitude.min() > -180.1 and read_data.longitude.max() < 180.1
        ), "Longitude values should be within the expected range for EPSG:4326"
        assert (
            read_data.latitude.min() > -90.1 and read_data.latitude.max() < 90.1
        ), "Latitude values should be within the expected range for EPSG:4326"
        assert (
            read_data.t2m.shape[1] < data.t2m.shape[1]
        ), "Longitude dimension should be smaller than original data due to geo clipping"
        assert (
            read_data.t2m.shape[0] < data.t2m.shape[0]
        ), "Latitude dimension should be smaller than original data due to geo clipping"
        assert (
            read_data.latitude.size == 13 and read_data.longitude.size == 9
        ), "Longitude and latitude dimensions should match the expected size after clipping"


def test_model_read_input_data_noclip(make_test_data, tmp_path):
    with make_test_data as data:
        model = jm.setup_modeldata(
            input=tmp_path / "test_data.nc",
            output="output_data.csv",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="parallelized",
            grid_data_baseurl=None,
            nuts_level=None,
            resolution=None,
            year=None,
        )
        read_data = jm.read_input_data(model).compute()
        assert isinstance(read_data, xr.Dataset), "should be xr dataset"
        assert "t2m" in read_data.data_vars, "correct data dim should be in the dataset"
        assert read_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"

        assert (
            read_data.longitude.min() > -180.1 and read_data.longitude.max() < 180.1
        ), "Longitude values should be within the expected range for EPSG:4326"
        assert (
            read_data.latitude.min() > -90.1 and read_data.latitude.max() < 90.1
        ), "Latitude values should be within the expected range for EPSG:4326"
        assert (
            read_data.latitude.size == data.latitude.size
            and read_data.longitude.size == data.longitude.size
        ), "Longitude dimension should the same  as input because of no clipping"
        assert not np.isnan(read_data.t2m.values).any()


def test_model_run(make_test_data, tmp_path):
    with make_test_data as _:  # only the written file is needed here
        model = jm.setup_modeldata(
            input=tmp_path / "test_data.nc",
            output=tmp_path / "output_data.nc",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="parallelized",
            grid_data_baseurl="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            nuts_level=3,
            resolution="10M",
            year=2024,
            out_colname="r0",
        )

        data = jm.read_input_data(model).compute()
        assert isinstance(data, xr.Dataset), "should be xr dataset"

        output_data = jm.run_model(model, data)
        assert isinstance(output_data, xr.DataArray), "should be xr dataset"

        jm.store_output_data(model, output_data)

        output_path = tmp_path / "output_data.nc"
        assert output_path.exists(), "Output file should be created"

        with xr.open_dataset(output_path) as data:
            output_data = data.compute()
            assert isinstance(output_data, xr.Dataset)
            assert "r0" in output_data.data_vars
            assert output_data.r0.shape == (
                13,
                9,
            ), "Output data shape should match input data shape"
            assert output_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"
            assert (
                output_data.latitude.min() >= -180.2
                and output_data.latitude.max() <= 180.2
            ), "Longitude values should be within the expected range for EPSG:4326"
            assert (
                output_data.longitude.min() >= -90.1
                and output_data.longitude.max() <= 90.2
            ), "Latitude values should be within the expected range for EPSG:4326"
            assert not np.isnan(
                output_data.r0.values
            ).all()  # TODO: this generates NaNs on occassion?!


def test_computation_with_default_config(tmp_path, make_test_data):
    with make_test_data as _:  # only the written file is needed here
        with open(
            Path.cwd() / "src" / "heiplanet_models" / "config_Jmodel.json", "r"
        ) as file:
            cfg = json.load(file)
            cfg["graph"]["setup_modeldata"]["kwargs"]["input"] = str(
                tmp_path / "test_data.nc",
            )
            cfg["graph"]["setup_modeldata"]["kwargs"]["output"] = str(
                tmp_path / "output_data2.nc"
            )
            cfg["graph"]["setup_modeldata"]["kwargs"]["r0_path"] = str(
                Path.cwd() / "test" / "test_r0.csv"
            )
        cgraph = cg.ComputationGraph(cfg)
        assert cgraph.config == cfg, "Config should be set correctly"

        cgraph.execute()

        output_path = tmp_path / "output_data2.nc"
        assert output_path.exists(), "Output file should be created"

        with xr.open_dataset(output_path) as data:
            output_data = data.compute()
            print(output_data)
            assert isinstance(output_data, xr.Dataset)
            assert "R0" in output_data.data_vars
            assert output_data.R0.shape == (
                13,
                9,
            ), "Output data shape should match input data shape"
            assert output_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"
            assert (
                output_data.latitude.min() >= -180.2
                and output_data.latitude.max() <= 180.2
            ), "Longitude values should be within the expected range for EPSG:4326"
            assert (
                output_data.longitude.min() >= -90.1
                and output_data.longitude.max() <= 90.2
            ), "Latitude values should be within the expected range for EPSG:4326"
            assert not np.isnan(output_data.R0.values).all()
