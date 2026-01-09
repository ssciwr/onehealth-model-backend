"""This script creates three dummy datasets (temperature, population, and rainfall)
with specified dimensions and variables for testing purposes. The datasets are saved
as NetCDF files in the specified directory.

The datasets created are:
1. Temperature Dataset: Contains temperature data with dimensions (time, latitude, longitude).
2. Population Dataset: Contains population density data with dimensions (time, latitude, longitude).
3. Rainfall Dataset: Contains rainfall data with dimensions (time, latitude, longitude).

Each dataset includes sample data and relevant metadata attributes.

The script also creates Octave-compatible versions of the same datasets with
different variable names to ensure compatibility with Octave-based processing.
The Octave-compatible datasets use the following variable names:
- Temperature Dataset: 't2m'
- Population Dataset: 'dens'
- Rainfall Dataset: 'tp'

The Octave-compatible datasets are also saved as NetCDF files in the specified directory.
"""

import numpy as np
import xarray as xr
import os
import logging


def create_dataset(
    var_name: str,
    lon: np.ndarray,
    lat: np.ndarray,
    time: np.ndarray,
    data: np.ndarray,
    coords_names: dict,
    out_path: str,
    attrs: dict,
) -> xr.Dataset:
    """
    Create and save a NetCDF dataset with given parameters.
    """
    ds = xr.Dataset(
        data_vars={
            var_name: (
                [coords_names["time"], coords_names["lat"], coords_names["lon"]],
                data,
            ),
        },
        coords={
            coords_names["lon"]: lon,
            coords_names["lat"]: lat,
            coords_names["time"]: time,
        },
        attrs=attrs,
    )
    if coords_names.get("transpose", False):
        ds = ds.transpose(
            coords_names["lon"], coords_names["lat"], coords_names["time"]
        )
    ds.to_netcdf(out_path)
    logging.info(f"Saved {out_path}")
    return ds


def main() -> None:
    """
    Main function to create dummy datasets for temperature, population, and rainfall.
    """
    logging.basicConfig(level=logging.INFO)
    out_dir = os.path.join("data", "in", "Pratik_datalake")
    os.makedirs(out_dir, exist_ok=True)

    # Common arrays
    lon1 = np.array([0, 1, 2], dtype=np.float64)
    lat1 = np.array([0, 1], dtype=np.float64)
    time1 = np.array(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
    )
    data1 = np.array(
        [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]],
            [[19, 20, 21], [22, 23, 24]],
        ],
        dtype=np.float64,
    )

    lon2 = np.array([0, 1, 2], dtype=np.float64)
    lat2 = np.array([0, 1], dtype=np.float64)
    time2 = np.array(["2024-01-01"], dtype="datetime64[ns]")
    data2 = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        ],
        dtype=np.float64,
    )

    lon3 = np.array([1, 2, 3], dtype=np.float64)
    lat3 = np.array([1, 2], dtype=np.float64)
    time3 = time1
    data3 = np.array(
        [
            [[-1, -2, -3], [-4, -5, -6]],
            [[-7, -8, -9], [-10, -11, -12]],
            [[-13, -14, -15], [-16, -17, -18]],
            [[-19, -20, -21], [-22, -23, -24]],
        ],
        dtype=np.float64,
    )

    # Dataset specifications
    datasets = [
        {
            "var_name": "temperature",
            "lon": lon1,
            "lat": lat1,
            "time": time1,
            "data": data1,
            "coords_names": {
                "lon": "longitude",
                "lat": "latitude",
                "time": "time",
                "transpose": True,
            },
            "out_file": "dataset_temperature_dummy.nc",
            "attrs": {
                "name": "Dummy Temperature",
                "description": "Dummy temperature dataset for testing purposes",
            },
        },
        {
            "var_name": "population",
            "lon": lon2,
            "lat": lat2,
            "time": time2,
            "data": data2,
            "coords_names": {
                "lon": "longitude",
                "lat": "latitude",
                "time": "time",
                "transpose": True,
            },
            "out_file": "dataset_population_dummy.nc",
            "attrs": {
                "name": "Dummy Population",
                "description": "Dummy population dataset for testing purposes",
            },
        },
        {
            "var_name": "rainfall",
            "lon": lon1,
            "lat": lat1,
            "time": time1,
            "data": data3,
            "coords_names": {
                "lon": "longitude",
                "lat": "latitude",
                "time": "time",
                "transpose": True,
            },
            "out_file": "dataset_rainfall_dummy.nc",
            "attrs": {
                "name": "Dummy Rainfall",
                "description": "Dummy rainfall dataset for testing purposes",
            },
        },
        # Octave-compatible datasets
        {
            "var_name": "t2m",
            "lon": lon1,
            "lat": lat1,
            "time": time1,
            "data": data1,
            "coords_names": {"lon": "longitude", "lat": "latitude", "time": "time"},
            "out_file": "dataset_temperature_dummy_octave.nc",
            "attrs": {
                "name": "Dummy Temperature",
                "description": "Dummy temperature dataset for testing purposes",
            },
        },
        {
            "var_name": "dens",
            "lon": lon2,
            "lat": lat2,
            "time": time2,
            "data": data2,
            "coords_names": {"lon": "lon", "lat": "lat", "time": "time"},
            "out_file": "dataset_population_dummy_octave.nc",
            "attrs": {
                "name": "Dummy Population",
                "description": "Dummy population dataset for testing purposes",
            },
        },
        {
            "var_name": "tp",
            "lon": lon1,
            "lat": lat1,
            "time": time1,
            "data": data3,
            "coords_names": {"lon": "longitude", "lat": "latitude", "time": "time"},
            "out_file": "dataset_rainfall_dummy_octave.nc",
            "attrs": {
                "name": "Dummy Rainfall",
                "description": "Dummy rainfall dataset for testing purposes",
            },
        },
        {
            "var_name": "dens",
            "lon": lon3,
            "lat": lat3,
            "time": time3,
            "data": data3,
            "coords_names": {"lon": "lon", "lat": "lat", "time": "time"},
            "out_file": "dataset_population_dummy_octave_notaligned.nc",
            "attrs": {
                "name": "Dummy Population not aligned",
                "description": "Dummy population dataset for testing purposes. This dataset has lon/lat not aligned with others.",
            },
        },
    ]

    for spec in datasets:
        out_path = os.path.join(out_dir, spec["out_file"])
        create_dataset(
            var_name=spec["var_name"],
            lon=spec["lon"],
            lat=spec["lat"],
            time=spec["time"],
            data=spec["data"],
            coords_names=spec["coords_names"],
            out_path=out_path,
            attrs=spec["attrs"],
        )


if __name__ == "__main__":
    main()
