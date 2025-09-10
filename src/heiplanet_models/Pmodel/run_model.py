
import logging 

from heiplanet_models.Pmodel.Pmodel_rates_development import carrying_capacity
from heiplanet_models.Pmodel.config import (
    PATH_ROOT,
    DATASET_PREFIXES,
    TIME_STEP,
)
from heiplanet_models.Pmodel.Pmodel_initial import load_data



INITIAL_YEAR = 2024
FINAL_YEAR = 2024


# Main processor
for year in range(INITIAL_YEAR, FINAL_YEAR + 1):
    logging.basicConfig(level=logging.DEBUG)
    # Generate filenames using f-strings for better readability
    path_file_temperature_dataset = PATH_ROOT / f'{DATASET_PREFIXES["TEMPERATURE_DATASET"]}{year}.nc'
    path_file_rainfall_dataset = PATH_ROOT / f'{DATASET_PREFIXES["RAINFALL_DATASET"]}{year}.nc'
    path_file_population_density_dataset = PATH_ROOT / f'{DATASET_PREFIXES["POPULATION_DENSITY_DATASET"]}{year}_global_0.5.nc'


    # Load variables
    print("--- Loading variables")
    model_data = load_data(
        path_temperature=path_file_temperature_dataset,
        path_rainfall=path_file_rainfall_dataset,
        path_population=path_file_population_density_dataset,
        time_step=TIME_STEP
    )
    print("--- Loading variables. DONE")

    print("--- Carrying capacity")
    CC = carrying_capacity(rainfall_data=model_data.rainfall, population_data=model_data.population_density)
    print("--- Carrying capacity. DONE")
    print(CC.shape)


    