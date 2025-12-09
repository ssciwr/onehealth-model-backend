"""
This script aligns the population dataset to match the temperature dataset's
coordinates and dimensions for further analysis. It create and store the aligned
population dataset in a new NetCDF file.
"""

import xarray as xr

TEMPERATURE_DATASET_PATH = (
    "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
)
POPULATION_DATASET_PATH = "data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc"


# 1. Load the ERA5Land temperature dataset
data_temperature = xr.open_dataset(TEMPERATURE_DATASET_PATH)
print(data_temperature)

# 2. Change the dimension names to standard ones in temperature dataset
data_temperature = data_temperature.rename({"t2m": "temperature", "valid_time": "time"})
print(data_temperature)
print(f"Sample longitudes: {data_temperature.longitude[:5].values}")
print(f"Sample latitudes: {data_temperature.latitude[:5].values}")

# 3. Load population dataset
data_population = xr.open_dataset(POPULATION_DATASET_PATH, decode_times=False)

# 4. Change the dimension names to standard ones in population dataset
data_population = data_population.rename(
    {"dens": "population", "lat": "latitude", "lon": "longitude"}
)
print(data_population)
print(f"Sample longitudes: {data_population.longitude[:5].values}")
print(f"Sample latitudes: {data_population.latitude[:5].values}")


# 5. Align coordinates and dimensions of data_population to match data_temperature
data_population = data_population.assign_coords(
    {"latitude": data_temperature.latitude, "longitude": data_temperature.longitude}
)
print(data_population.longitude[:5].values)
print(data_population.latitude[:5].values)


# 6. Remove the existing time dimension from data_population
data_population = data_population.isel(time=0).drop_vars("time")

# Expand data_population along the new time dimension
data_population_broadcasted = data_population.expand_dims(
    {"time": [data_temperature.time.isel(time=0).values]}
)

print(data_population_broadcasted)

# result = data_temperature["temperature"] + data_population_broadcasted["population"]

# print(result.values)

data_population_broadcasted.to_netcdf(
    "data/in/Pratik_datalake/pop_dens_2024_global_0.5_aligned.nc"
)
