"""This script creates a dummy initial conditions dataset as an xarray.DataArray
with dimensions (3, 2, 5) and saves it as a NetCDF file. The structure and style
follow create_dummy_datasets.py for consistency.
"""

import numpy as np
import xarray as xr
import os
import logging


def create_initial_conditions_datasets(
    out_dir: str,
    attrs: dict = None,
):
    """
    Create and save dummy initial conditions NetCDF files for both Python and Octave conventions.
    """
    if attrs is None:
        attrs = {
            "name": "Dummy Initial Conditions",
            "description": "Dummy initial conditions array for testing purposes",
        }

    # Provided data (shape (3,2,4))
    data_base = np.array(
        [
            [[10.0, 10.6, 11.2, 11.8], [10.3, 10.9, 11.5, 12.1]],
            [[10.1, 10.7, 11.3, 11.9], [10.4, 11.0, 11.6, 12.2]],
            [[10.2, 10.8, 11.4, 12.0], [10.5, 11.1, 11.7, 12.3]],
        ],
        dtype=np.float64,
    )

    # Coordinates for (3,2,4)
    longitude = np.arange(3)
    latitude = np.arange(2)
    time = np.arange(4)

    # Transpose to (time, latitude, longitude)
    data_base_t = np.transpose(data_base, (2, 1, 0))

    # Python-style dataset
    ds_py = xr.Dataset(
        data_vars={
            "eggs": (["longitude", "latitude", "time"], data_base * 0.5),
            "ed": (["longitude", "latitude", "time"], data_base * 1),
            "juv": (["longitude", "latitude", "time"], data_base * 1.5),
            "imm": (["longitude", "latitude", "time"], data_base * 2.0),
            "adults": (["longitude", "latitude", "time"], data_base * 3.0),
        },
        coords={
            "longitude": ("longitude", longitude),
            "latitude": ("latitude", latitude),
            "time": ("time", time),
        },
        attrs=attrs,
    )
    out_path_py = os.path.join(out_dir, "initial_conditions_dummy.nc")
    ds_py.to_netcdf(out_path_py)
    logging.info(f"Saved {out_path_py}")

    # Octave-style dataset: use variable names and dimension order compatible with Octave
    # (e.g., dims: longitude, latitude, time; variable names: eggs, ed, juv, imm, adults)
    # We'll use the original data_base (3,2,4) and dims order: longitude, latitude, time
    ds_oct = xr.Dataset(
        data_vars={
            "eggs": (["time", "latitude", "longitude"], data_base_t * 0.5),
            "ed": (["time", "latitude", "longitude"], data_base_t * 1),
            "juv": (["time", "latitude", "longitude"], data_base_t * 1.5),
            "imm": (["time", "latitude", "longitude"], data_base_t * 2.0),
            "adults": (["time", "latitude", "longitude"], data_base_t * 3.0),
        },
        coords={
            "longitude": ("longitude", longitude),
            "latitude": ("latitude", latitude),
            "time": ("time", time),
        },
        attrs=attrs,
    )
    out_path_oct = os.path.join(out_dir, "initial_conditions_dummy_octave.nc")
    ds_oct.to_netcdf(out_path_oct)
    logging.info(f"Saved {out_path_oct}")
    return ds_py, ds_oct


def main():
    logging.basicConfig(level=logging.INFO)
    out_dir = os.path.join("test", "resources")
    os.makedirs(out_dir, exist_ok=True)
    create_initial_conditions_datasets(out_dir=out_dir)


if __name__ == "__main__":
    main()
