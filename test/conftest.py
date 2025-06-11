import pytest
import numpy as np
import xarray as xr
from pathlib import Path


def make_rioxarray_testdata(path: Path, valid: bool = True, resolution: int = 10) -> xr.Dataset:
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
