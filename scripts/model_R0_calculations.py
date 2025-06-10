import os

import pandas as pd
import xarray as xr

from model_backend.interpolators import interpolator_function
from model_backend.r0_calculations import R0_pip_fun
from model_backend.read_geodata import (
    read_nuts_data,
    read_netcdf4_format,
)
from model_backend.write_geodata import write_to_geotiff


PATH_DATA_IN = "./data/in/"
PATH_DATA_OUT = "./data/out/"

FILENAME_NUTS = "NUTS_RG_10M_2024_4326.shp/NUTS_RG_10M_2024_4326.shp"
FILENAME_ERA5LAND = "ERA5land_global_t2m_dailyStats_mean_01Deg_2024_08_data.nc"
FILENAME_R0_POSTERIOR_STATS = "R0_pip_stats.csv"


path_file_nuts = os.path.join(PATH_DATA_IN, FILENAME_NUTS)
path_file_era5land = os.path.join(PATH_DATA_IN, FILENAME_ERA5LAND)
path_file_r0_posterior_stats = os.path.join(PATH_DATA_IN, FILENAME_R0_POSTERIOR_STATS)


# Approved
def create_europa_bbox(nuts_dataset):
    minx, miny, maxx, maxy = nuts_dataset.total_bounds
    europe_bounds_nuts = {
        "min_longitude": minx,
        "min_latitude": miny,
        "max_longitude": maxx,
        "max_latitude": maxy,
    }

    return europe_bounds_nuts


# Approved
def crop_raster(netcdf4_dataset, nuts_dataset):
    # Extract bounding box from nuts dataset
    bounding_box = create_europa_bbox(nuts_dataset)

    # Generate slices for latitude and longitude based on the bounding box
    latitude_raster = netcdf4_dataset.latitude.values
    is_lat_descending = latitude_raster[0] > latitude_raster[-1]
    print(f"Descending latitude: {is_lat_descending}")

    latitude_slice = (
        slice(bounding_box["max_latitude"], bounding_box["min_latitude"])
        if is_lat_descending
        else slice(bounding_box["min_latitude"], bounding_box["max_latitude"])
    )

    longitude_slice = slice(
        bounding_box["min_longitude"], bounding_box["max_longitude"]
    )

    # Crop the netcdf4 dataset based on the bounding box
    cropped_ds = netcdf4_dataset.sel(latitude=latitude_slice, longitude=longitude_slice)

    return cropped_ds


# Approved
def create_interpolator():

    X_COLUMN_STATS = "Temperature"
    Y_COLUMN_STATS = "Median_R0"

    # Read the posterior stats
    df = pd.read_csv(path_file_r0_posterior_stats)

    # Return the interpolator based on the posterior stats
    try:
        return interpolator_function(
            x_data=df[X_COLUMN_STATS],
            y_data=df[Y_COLUMN_STATS],
            method="linear",
        )
    except Exception as e:
        print(e)


def define_ranges_processor(initial_year, final_year, initial_month, final_month):
    years_to_process = range(initial_year, final_year + 1, 1)
    months_to_process = range(initial_month, final_month + 1, 1)

    return years_to_process, months_to_process


def main():

    # 1. Define the range of years to process
    initial_year = 2023
    final_year = 2024

    # 2. Defin the range of months to process
    initial_month = 8
    final_month = 8

    years_to_process, months_to_process = define_ranges_processor(
        initial_year,
        final_year,
        initial_month,
        final_month,
    )

    # TODO: Read NUTS dataset
    nuts_ds = read_nuts_data(path_file=path_file_nuts)

    # Invoke pre-build interpolator
    interp_fun = create_interpolator()

    for year in years_to_process:
        for month in months_to_process:
            # TODO: Read ERA-5 Land
            filename_era5land = (
                f"ERA5land_global_t2m_dailyStats_mean_01Deg_{year}_{month:02d}_data.nc"
            )
            era5land_ds = read_netcdf4_format(
                path_file=os.path.join(PATH_DATA_IN, filename_era5land)
            )

            # TODO: Crop ERA5 land according to the NUTS data.
            cropped = crop_raster(
                netcdf4_dataset=era5land_ds,
                nuts_dataset=nuts_ds,
            )

            # TODO: Apply the interpolation function to the cropped ERA5land
            R0_posterior = xr.DataArray(
                R0_pip_fun(cropped["t2m"].values, interpolator=interp_fun),
                dims=cropped["t2m"].dims,
                coords=cropped["t2m"].coords,
                attrs=cropped["t2m"].attrs,
            )

            print(R0_posterior.dims)

            # TODO: Save processed raster(cropped era5) in .tif format
            dict_tiff = {"band": "valid_time", "x": "longitude", "y": "latitude"}
            output_filename = f"R0_pip_posterior_median_europe_{year}_{month:02d}.tif"
            write_to_geotiff(
                R0_posterior,
                path_file=os.path.join(PATH_DATA_OUT, output_filename),
                dict_variables=dict_tiff,
            )


if __name__ == "__main__":
    main()
