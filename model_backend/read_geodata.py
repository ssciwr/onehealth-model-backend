"""This module contains functions to read the following formats
- NETCDF4
- .shp for NUTS dataset
"""

import logging
import geopandas as gpd
import xarray as xr
from pathlib import Path
from typing import Optional


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


def read_nuts_data(path_file: str) -> Optional[gpd.GeoDataFrame]:
    """
    Reads a geospatial file (e.g., shapefile, GeoJSON) into a GeoDataFrame.

    Args:
        path_file (str): Path to the geospatial file.

    Returns:
        Optional[geopandas.GeoDataFrame]: GeoDataFrame if successful, otherwise None.
    """
    path = Path(path_file)
    if not path.exists():
        logging.warning(f"File does not exist: {path}")
        return None
    if path.stat().st_size == 0:
        logging.warning(f"File is empty: {path}")
        return None

    try:
        nuts_gdf = gpd.read_file(path)
        logging.info(f"Successfully loaded geospatial data from {path}")
        return nuts_gdf
    except Exception as e:
        logging.error(f"Error reading geospatial file {path}: {e}")
        return None
