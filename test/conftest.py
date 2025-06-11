import pytest
import numpy as np
import xarray as xr
from pathlib import Path
import rioxarray  # noqa Make sure this is imported


def make_rioxarray_testdata(path: Path, valid: bool = True) -> xr.Dataset:
    if valid:
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)
    else:
        lon = np.linspace(-120, 120, 10)
        lat = np.linspace(-70, 76, 10)
    data = np.random.rand(10, 10)

    # Create dataset with proper dimension order for geospatial data
    ds = xr.Dataset(
        data_vars={
            "t2m": (("y", "x"), data),
        },
        coords={
            "y": lat,
            "x": lon,
        },
    )

    # Save to netCDF
    ds.to_netcdf(path / "test_data.nc", mode="w", format="NETCDF4")

    # First load with netCDF engine
    with xr.open_dataset(path / "test_data.nc") as data:
        # Properly set spatial attributes needed by rioxarray - this is what makes .rio work
        ds = data.copy()  # FIXME: this still leaks resources
    return ds


@pytest.fixture
def make_test_data(tmp_path) -> xr.Dataset:
    return make_rioxarray_testdata(tmp_path, valid=True)


@pytest.fixture
def make_invalid_test_data(tmp_path) -> xr.Dataset:
    return make_rioxarray_testdata(tmp_path, valid=False)
