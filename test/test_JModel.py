from model_backend import JModel
import xarray as xr
from pathlib import Path
import pandas as pd
import pytest
import math
import numpy as np


def test_jmodel_initialization():
    # Test valid initialization

    path = Path.cwd() / "test" / "test_r0.csv"

    model = JModel(
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
        JModel(
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
        JModel(
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
        JModel(
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
        JModel(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=path,
            run_mode="parallelized",  # Invalid run mode
            grid_data_baseurl=None,
            nuts_level=3,
            resolution="10M",
            year=2024,
        )


def test_model_read_input_data(make_test_data, tmp_path):
    with make_test_data as data:
        model = JModel(
            input=tmp_path / "test_data.nc",
            output="output_data.csv",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="parallelized",
            grid_data_baseurl="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

        read_data = model.read_input_data().compute()

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
        model = JModel(
            input=tmp_path / "test_data.nc",
            output="output_data.csv",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="parallelized",
            grid_data_baseurl=None,
            nuts_level=None,
            resolution=None,
            year=None,
        )
        read_data = model.read_input_data().compute()
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
        model = JModel(
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

        model.run()
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
            assert not np.isnan(output_data.r0.values).all()
