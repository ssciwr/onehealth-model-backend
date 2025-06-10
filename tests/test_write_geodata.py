import pytest
import numpy as np
import xarray as xr

from model_backend.write_geodata import write_to_geotiff


@pytest.fixture
def simple_xarray_2d(tmp_path):
    data = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("latitude", "longitude"),
        coords={"latitude": [10, 20], "longitude": [30, 40, 50]},
        name="test",
    )
    return data


@pytest.fixture
def simple_xarray_3d(tmp_path):
    data = xr.DataArray(
        np.arange(12).reshape(2, 2, 3),
        dims=("band", "latitude", "longitude"),
        coords={"band": [1, 2], "latitude": [10, 20], "longitude": [30, 40, 50]},
        name="test",
    )
    return data


def test_write_to_geotiff_2d(tmp_path, simple_xarray_2d):
    path = tmp_path / "test2d.tif"
    dict_vars = {
        "x": "longitude",
        "y": "latitude",
    }  # band not used for 2D
    write_to_geotiff(simple_xarray_2d, str(path), dict_vars)
    assert path.exists()
    assert path.stat().st_size > 0


def test_write_to_geotiff_3d(tmp_path, simple_xarray_3d):
    path = tmp_path / "test3d.tif"
    dict_vars = {"x": "longitude", "y": "latitude", "band": "band"}
    write_to_geotiff(simple_xarray_3d, str(path), dict_vars)
    assert path.exists()
    assert path.stat().st_size > 0


def test_write_to_geotiff_invalid_dim(tmp_path):
    data = xr.DataArray(np.arange(4), dims=("a",), coords={"a": [1, 2, 3, 4]})
    path = tmp_path / "invalid.tif"
    dict_vars = {"x": "a", "y": "a", "band": "a"}
    with pytest.raises(ValueError):
        write_to_geotiff(data, str(path), dict_vars)
