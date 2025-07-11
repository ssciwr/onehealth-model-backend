import xarray as xr

import Pmodel_input
from Pmodel_initial import (
    load_temperature,
    load_human_population_density,
    load_rainfall,
    load_latitude,
    # Formula
    juvenile_carrying_capacity,
)

filepath_dataset_temperature = "/home/ecarreno/SSC-Projects/b_REPOSITORIES/onehealth-model-backend/data/in/ERA5land_global_t2m_daily_0.5_2024.nc"
filepath_dataset_rainfall = "/home/ecarreno/SSC-Projects/b_REPOSITORIES/onehealth-model-backend/data/in/ERA5land_global_tp_daily_0.5_2024.nc"
filepath_dataset_human_population = "/home/ecarreno/SSC-Projects/b_REPOSITORIES/onehealth-model-backend/data/in/pop_dens_2024_global_0.5.nc"


VAR_TEMPERATURE = "t2m"
VAR_RAINFALL = "tp"
VAR_HUMAN_POPULATION = "total-population"

STEP_SIZE = 10

input_data = Pmodel_input.PmodelInput(
    var_temperature=xr.open_dataset(filepath_dataset_temperature)[VAR_TEMPERATURE],
    var_rainfall=xr.open_dataset(filepath_dataset_rainfall)[VAR_RAINFALL],
    var_human_population=xr.open_dataset(filepath_dataset_human_population)[
        VAR_HUMAN_POPULATION
    ],
)


temperature, temperature_mean = load_temperature(input_data.var_temperature, STEP_SIZE)
population_density = load_human_population_density(input_data.var_human_population)
rainfall = load_rainfall(input_data.var_rainfall)
latitude = load_latitude(
    filepath_dataset=filepath_dataset_temperature, variable_name="latitude"
)

carrying_capacity = juvenile_carrying_capacity(rainfall, population_density)


print(f"Dim temperature: {temperature.shape}")
print(f"Dim temperature_mean: {temperature_mean.shape}")
print(f"Dim population_density: {population_density.shape}")

print(f"Dim rainfall: {rainfall.shape}")
print(f"Dim latitude: {latitude.shape}")
# print(f"Dim. carrying capacity: {carrying_capacity.shape}")

# print(f"{population_density.coords}")
