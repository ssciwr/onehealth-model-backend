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

# ----------------------------------------------------------------
# ------------------   Dataset 1  (Temperature)  -----------------
# ----------------------------------------------------------------
# Three longitudes
lon1 = np.array([0, 1, 2], dtype=np.float64)

# Two latitudes
lat1 = np.array([0, 1], dtype=np.float64)

# Four timestamps
time1 = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
)

# shape (4, 2, 3) --> dimensions [time, latitude, longitude]
data_array_1 = np.array(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [13, 14, 15],
            [16, 17, 18],
        ],
        [
            [19, 20, 21],
            [22, 23, 24],
        ],
    ],
    dtype=np.float64,
)

# data_array_1 = np.random.randint(1, 61, size=(3, 4, 5))

dataset_temperature = xr.Dataset(
    data_vars={
        "temperature": (["time", "latitude", "longitude"], data_array_1),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
    attrs={
        "name": "Dummy Temperature",
        "description": "Dummy temperature dataset for testing purposes",
    },
)

# Transpose dimensions to (longitude, latitude, time)
dataset_temperature = dataset_temperature.transpose("longitude", "latitude", "time")

# Example [time, latitude, longitude]
dataset_temperature_slice = dataset_temperature.temperature[0, :, :]
print(dataset_temperature_slice.values)

# Write to netcdf file
dataset_temperature.to_netcdf("data/in/Pratik_datalake/dataset_temperature_dummy.nc")


# ----------------------------------------------------------------
# ----------------     Dataset 2  (Population)    ----------------
# ----------------------------------------------------------------

# Five longitudes
lon2 = np.array([0, 1, 2], dtype=np.float64)

# Four latitudes
lat2 = np.array([0, 1], dtype=np.float64)

# One timestamp
time2 = np.array(["2024-01-01"], dtype="datetime64[ns]")
data_array_2 = np.array(
    [
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    ],
    dtype=np.float64,
)

# data_array_2 = np.random.randint(1, 61, size=(1, 2, 3))

dataset_population = xr.Dataset(
    data_vars={
        "population": (["time", "latitude", "longitude"], data_array_2),
    },
    coords={
        "longitude": lon2,
        "latitude": lat2,
        "time": time2,
    },
    attrs={
        "name": "Dummy Population",
        "description": "Dummy population dataset for testing purposes",
    },
)

# Transpose dimensions to (longitude, latitude, time)
dataset_population = dataset_population.transpose("longitude", "latitude", "time")

# Example
dataset_population_slice = dataset_population.population[0, :, :]
print(dataset_population_slice.values)


# Write to netcdf file
dataset_population.to_netcdf("data/in/Pratik_datalake/dataset_population_dummy.nc")


# -------------------------------------------------------------
# ------------------   Dataset 3  (rainfall)  -----------------
# -------------------------------------------------------------
# Three longitudes
lon3 = np.array([1, 2, 3], dtype=np.float64)

# Two latitudes
lat3 = np.array([1, 2], dtype=np.float64)

# Four timestamps
time3 = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
)

# shape (4, 2, 3) --> dimensions [time, latitude, longitude]
data_array_3 = np.array(
    [
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        [
            [-7, -8, -9],
            [-10, -11, -12],
        ],
        [
            [-13, -14, -15],
            [-16, -17, -18],
        ],
        [
            [-19, -20, -21],
            [-22, -23, -24],
        ],
    ],
    dtype=np.float64,
)

# data_array_3 = np.random.randint(1, 61, size=(3, 4, 5))

dataset_rainfall = xr.Dataset(
    data_vars={
        "rainfall": (["time", "latitude", "longitude"], data_array_3),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
    attrs={
        "name": "Dummy Rainfall",
        "description": "Dummy rainfall dataset for testing purposes",
    },
)

# Transpose dimensions to (longitude, latitude, time)
dataset_rainfall = dataset_rainfall.transpose("longitude", "latitude", "time")
print(dataset_rainfall.dims)

# Example [time, latitude, longitude]
dataset_rainfall_slice = dataset_rainfall.rainfall[0, :, :]
print(dataset_rainfall_slice.values)

# Write to netcdf file
dataset_rainfall.to_netcdf("data/in/Pratik_datalake/dataset_rainfall_dummy.nc")


# ----------------------------------------------------------------


# ----------------------------------------------------------------------
# ------------------   Dataset 1 Octave (Temperature)  -----------------
# ----------------------------------------------------------------------
# Three longitudes
lon1 = np.array([0, 1, 2], dtype=np.float64)

# Two latitudes
lat1 = np.array([0, 1], dtype=np.float64)

# Four timestamps
time1 = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
)

# shape (4, 2, 3) --> dimensions [time, latitude, longitude]
data_array_1 = np.array(
    [
        [
            [1, 2, 3],
            [4, 5, 6],
        ],
        [
            [7, 8, 9],
            [10, 11, 12],
        ],
        [
            [13, 14, 15],
            [16, 17, 18],
        ],
        [
            [19, 20, 21],
            [22, 23, 24],
        ],
    ],
    dtype=np.float64,
)

# data_array_1 = np.random.randint(1, 61, size=(3, 4, 5))

dataset_temperature = xr.Dataset(
    data_vars={
        "t2m": (["time", "latitude", "longitude"], data_array_1),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
    attrs={
        "name": "Dummy Temperature",
        "description": "Dummy temperature dataset for testing purposes",
    },
)

# Example [time, latitude, longitude]
dataset_temperature_slice = dataset_temperature.t2m[0, :, :]
print(dataset_temperature_slice.values)

# Write to netcdf file
dataset_temperature.to_netcdf(
    "data/in/Pratik_datalake/dataset_temperature_dummy_octave.nc"
)


# ----------------------------------------------------------------
# ----------------     Dataset 2  (Population)    ----------------
# ----------------------------------------------------------------

# Five longitudes
lon2 = np.array([0, 1, 2], dtype=np.float64)

# Four latitudes
lat2 = np.array([0, 1], dtype=np.float64)

# One timestamp
time2 = np.array(["2024-01-01"], dtype="datetime64[ns]")
data_array_2 = np.array(
    [
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    ],
    dtype=np.float64,
)

# data_array_2 = np.random.randint(1, 61, size=(1, 2, 3))

dataset_population = xr.Dataset(
    data_vars={
        "dens": (["time", "lat", "lon"], data_array_2),
    },
    coords={
        "lon": lon2,
        "lat": lat2,
        "time": time2,
    },
    attrs={
        "name": "Dummy Population",
        "description": "Dummy population dataset for testing purposes",
    },
)

# Example
dataset_population_slice = dataset_population.dens[0, :, :]
print(dataset_population_slice.values)


# Write to netcdf file
dataset_population.to_netcdf(
    "data/in/Pratik_datalake/dataset_population_dummy_octave.nc"
)


# -------------------------------------------------------------
# ------------------   Dataset 3  (rainfall)  -----------------
# -------------------------------------------------------------
# Three longitudes
lon3 = np.array([1, 2, 3], dtype=np.float64)

# Two latitudes
lat3 = np.array([1, 2], dtype=np.float64)

# Four timestamps
time3 = np.array(
    ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"], dtype="datetime64[ns]"
)

# shape (4, 2, 3) --> dimensions [time, latitude, longitude]
data_array_3 = np.array(
    [
        [
            [-1, -2, -3],
            [-4, -5, -6],
        ],
        [
            [-7, -8, -9],
            [-10, -11, -12],
        ],
        [
            [-13, -14, -15],
            [-16, -17, -18],
        ],
        [
            [-19, -20, -21],
            [-22, -23, -24],
        ],
    ],
    dtype=np.float64,
)

# data_array_3 = np.random.randint(1, 61, size=(3, 4, 5))

dataset_rainfall = xr.Dataset(
    data_vars={
        "tp": (["time", "latitude", "longitude"], data_array_3),
    },
    coords={
        "longitude": lon1,
        "latitude": lat1,
        "time": time1,
    },
    attrs={
        "name": "Dummy Rainfall",
        "description": "Dummy rainfall dataset for testing purposes",
    },
)

# Example [time, latitude, longitude]
dataset_rainfall_slice = dataset_rainfall.tp[0, :, :]
print(dataset_rainfall_slice.values)

# Write to netcdf file
dataset_rainfall.to_netcdf("data/in/Pratik_datalake/dataset_rainfall_dummy_octave.nc")
