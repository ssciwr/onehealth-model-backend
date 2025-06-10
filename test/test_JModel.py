from model_backend import JModel
from pathlib import Path
import pandas as pd
import pytest
import xarray as xr
import math


def test_jmodel_initialization():
    # Test valid initialization

    path = Path.cwd() / "test" / "test_r0.csv"

    model = JModel(
        input="input_data.csv",
        output="output_data.csv",
        r0_path=path,
        run_mode="allowed",
        grid_data_baseurl="https://example.com/grid_data",
        nuts_level=3,
        resolution="10M",
        year=2024,
    )

    assert model.input == "input_data.csv"
    assert model.output == "output_data.csv"
    assert model.run_mode == "allowed"
    assert isinstance(model.r0_data, pd.DataFrame)
    assert model.grid_data_baseurl == "https://example.com/grid_data"
    assert model.nuts_level == 3
    assert model.resolution == "10M"
    assert model.year == 2024
    assert math.isclose(model.min_temp, 8.6)
    assert math.isclose(model.max_temp, 13.0)

    with pytest.raises(ValueError):
        JModel(
            input=None,
            output="output_data.csv",
            r0_path=path,
            run_mode="invalid_forbiddenmode",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )
    with pytest.raises(ValueError):
        JModel(
            input="input_data.csv",
            output=None,
            r0_path=path,
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
            r0_path=path,
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
            run_mode="parallel",  # Invalid run mode
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
            run_mode="allowed",
            grid_data_baseurl="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )
        read_data = model.read_input_data()
        assert isinstance(read_data, xr.Dataset)
        assert "t2m" in read_data.data_vars
        assert read_data.t2m == data.t2m
        assert read_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"

        assert read_data.x.min() > -180.1 and read_data.x.max() < 180.1
        assert read_data.y.min() > -90.1 and read_data.y.max() < 90.1
        assert (
            read_data.x.size == 1 and read_data.y.size == 1
        )  # due to clipping on the european union and a very coarse grid we have for test data
        assert read_data.t2m[0, 0] == data.t2m.values[7, 5], (
            "Temperature data should match the test data values where it's clipped"
        )


def test_model_run(make_test_data, tmp_path):
    with make_test_data as _:
        model = JModel(
            input=tmp_path / "test_data.nc",
            output=tmp_path / "output_data.nc",
            r0_path=Path.cwd() / "test" / "test_r0.csv",
            run_mode="allowed",
            grid_data_baseurl="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

        model.run()
        output_path = tmp_path / "output_data.nc"
        assert output_path.exists(), "Output file should be created"

        with xr.open_dataset(output_path) as output_data:
            assert isinstance(output_data, xr.Dataset)
            assert "t2m" in output_data.data_vars
            assert output_data.t2m.shape == (1, 1), (
                "Output data shape should match input data shape"
            )
            assert output_data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"
            assert output_data.x.min() > -180.1 and output_data.x.max() < 180.1, (
                "Longitude values should be within the expected range for EPSG:4326"
            )
            assert output_data.y.min() >= -90.1 and output_data.y.max() <= 90.1, (
                "Latitude values should be within the expected range for EPSG:4326"
            )
