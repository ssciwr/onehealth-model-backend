"""This module contains functions to read the following formats
- NETCDF4
-
"""

import logging
from pathlib import Path
import xarray as xr
from typing import Optional
import geopandas as gpd

logging.basicConfig(level=logging.INFO)


def read_netcdf4_format(path_file: str) -> Optional[xr.Dataset]:
    """Read NETCDF4 files as xarray datasets.

    Args:
        path_file (str): Path to the NETCDF4 file.

    Returns:
        Optional[xr.Dataset]: xarray dataset if successful, otherwise None.
    """
    path = Path(path_file)

    if not path.exists():
        logging.warning(f"File does not exist: {path}")
        return None
    if path.suffix.lower() != ".nc":
        logging.warning(f"File is not a NetCDF (.nc) file: {path}")
        return None
    if path.stat().st_size == 0:
        logging.warning(f"File is empty: {path}")
        return None

    try:
        xarray_ds = xr.open_dataset(path, engine="netcdf4")
        logging.info(
            f"Function read_netcdf4_format({path_file}) has been successfully loaded."
        )
        return xarray_ds
    except Exception as e:
        logging.error(f"Error opening NetCDF4 file {path}: {e}")
        return None


def get_eurostat_geospatial(
    nuts_level=3,
    year=2024,
    resolution="10M",
    base_url="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
    url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
):
    # TODO: check again that this works as intended. eurostat has no good python package anymore.
    # alternatively use r2py and the r package instead

    # Available formats: geojson, shp, topojson
    url_str = url(
        nuts_level=nuts_level, year=year, resolution=resolution, base_url=base_url
    )

    try:
        nuts_data = gpd.read_file(url_str)
        return nuts_data
    except Exception as e:
        print(f"Failed to download from {url}: {e}")
        return None
