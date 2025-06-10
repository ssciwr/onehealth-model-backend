import xarray as xr
import numpy as np


def make_test_data(tmp_path, defect=False) -> xr.Dataset:
    if defect:
        lon = np.linspace(-120, 120, 10)
        lat = np.linspace(-70, 76, 10)
    else:
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)

    # Create the dataset with integer indices as coordinates
    data = np.random.rand(10, 10)  # Use y,x order for data array

    ds = xr.Dataset(
        data_vars={
            "temp": (("y", "x"), data),  # Use y,x order for dims
        },
        coords={
            "y": lon,  # Integer indices
            "x": lat,  # Integer indices
        },
    )

    ds.to_netcdf(tmp_path / "test_data.nc")
    ds = xr.open_dataset(tmp_path / "test_data.nc", engine="rasterio")
    return ds
