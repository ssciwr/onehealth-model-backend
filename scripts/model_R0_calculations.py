"""
Script for calculating R0 posterior rasters for Europe using climate and geospatial data.

This script reads NUTS boundaries, ERA5-Land temperature data, and R0 posterior statistics,
applies interpolation, and writes the results as GeoTIFF files.
"""

import os
import logging
from typing import Any, Dict, Tuple

import pandas as pd
import xarray as xr

from model_backend.interpolators import interpolator_function
from model_backend.r0_calculations import R0_pip_fun
from model_backend.read_geodata import (
    read_nuts_data,
    read_netcdf4_format,
)
from model_backend.write_geodata import write_to_geotiff

# Configure logging
logging.basicConfig(level=logging.INFO)

PATH_DATA_IN = "./data/in/"
PATH_DATA_OUT = "./data/out/"

FILENAME_NUTS = "NUTS_RG_10M_2024_4326.shp/NUTS_RG_10M_2024_4326.shp"
FILENAME_R0_POSTERIOR_STATS = "R0_pip_stats.csv"

path_file_nuts = os.path.join(PATH_DATA_IN, FILENAME_NUTS)
path_file_r0_posterior_stats = os.path.join(PATH_DATA_IN, FILENAME_R0_POSTERIOR_STATS)


def create_europa_bbox(nuts_dataset: Any) -> Dict[str, float]:
    """
    Create a bounding box dictionary from a NUTS dataset.

    Args:
        nuts_dataset: Geopandas GeoDataFrame with total_bounds attribute.

    Returns:
        dict: Bounding box with min/max longitude and latitude.
    """
    minx, miny, maxx, maxy = nuts_dataset.total_bounds
    return {
        "min_longitude": minx,
        "min_latitude": miny,
        "max_longitude": maxx,
        "max_latitude": maxy,
    }


def crop_raster(netcdf4_dataset: xr.Dataset, nuts_dataset: Any) -> xr.Dataset:
    """
    Crop a NetCDF4 dataset to the bounding box of the NUTS dataset.

    Args:
        netcdf4_dataset: xarray Dataset with latitude and longitude.
        nuts_dataset: Geopandas GeoDataFrame.

    Returns:
        xr.Dataset: Cropped dataset.
    """
    bounding_box = create_europa_bbox(nuts_dataset)
    latitude_raster = netcdf4_dataset.latitude.values
    is_lat_descending = latitude_raster[0] > latitude_raster[-1]
    logging.info(f"Descending latitude: {is_lat_descending}")

    latitude_slice = (
        slice(bounding_box["max_latitude"], bounding_box["min_latitude"])
        if is_lat_descending
        else slice(bounding_box["min_latitude"], bounding_box["max_latitude"])
    )
    longitude_slice = slice(
        bounding_box["min_longitude"], bounding_box["max_longitude"]
    )
    return netcdf4_dataset.sel(latitude=latitude_slice, longitude=longitude_slice)


def create_interpolator() -> Any:
    """
    Create an interpolation function from R0 posterior statistics.

    Returns:
        Callable: Interpolator function.
    """
    X_COLUMN_STATS = "Temperature"
    Y_COLUMN_STATS = "Median_R0"
    try:
        df = pd.read_csv(path_file_r0_posterior_stats)
        return interpolator_function(
            x_data=df[X_COLUMN_STATS],
            y_data=df[Y_COLUMN_STATS],
            method="linear",
        )
    except Exception as e:
        logging.error(f"Error creating interpolator: {e}")
        raise


def define_ranges_processor(
    initial_year: int, final_year: int, initial_month: int, final_month: int
) -> Tuple[range, range]:
    """
    Define the range of years and months to process.

    Args:
        initial_year (int): Start year.
        final_year (int): End year.
        initial_month (int): Start month.
        final_month (int): End month.

    Returns:
        Tuple[range, range]: Ranges for years and months.
    """
    years_to_process = range(initial_year, final_year + 1)
    months_to_process = range(initial_month, final_month + 1)
    return years_to_process, months_to_process


def main() -> None:
    """
    Main processing function for R0 calculation and raster export.
    """
    initial_year = 2023
    final_year = 2024
    initial_month = 8
    final_month = 8

    years_to_process, months_to_process = define_ranges_processor(
        initial_year, final_year, initial_month, final_month
    )

    try:
        nuts_ds = read_nuts_data(path_file=path_file_nuts)
        interp_fun = create_interpolator()
    except Exception as e:
        logging.error(f"Initialization failed: {e}")
        return

    for year in years_to_process:
        logging.info("----------------------------")
        logging.info(f"---- Processing year: {year}")
        for month in months_to_process:
            filename_era5land = (
                f"ERA5land_global_t2m_dailyStats_mean_01Deg_{year}_{month:02d}_data.nc"
            )
            era5land_path = os.path.join(PATH_DATA_IN, filename_era5land)
            try:
                era5land_ds = read_netcdf4_format(path_file=era5land_path)
                cropped = crop_raster(era5land_ds, nuts_ds)
                R0_posterior = xr.DataArray(
                    R0_pip_fun(cropped["t2m"].values, interpolator=interp_fun),
                    dims=cropped["t2m"].dims,
                    coords=cropped["t2m"].coords,
                    attrs=cropped["t2m"].attrs,
                )
                dict_tiff = {"band": "valid_time", "x": "longitude", "y": "latitude"}
                output_filename = (
                    f"R0_pip_posterior_median_europe_{year}_{month:02d}.tif"
                )
                write_to_geotiff(
                    R0_posterior,
                    path_file=os.path.join(PATH_DATA_OUT, output_filename),
                    dict_variables=dict_tiff,
                )
                logging.info(f"Processed and saved: {output_filename}")
            except Exception as e:
                logging.error(f"Failed for {filename_era5land}: {e}")


if __name__ == "__main__":
    main()
