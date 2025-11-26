import logging

from heiplanet_models.Pmodel.Pmodel_rates_development import carrying_capacity
from heiplanet_models.Pmodel.Pmodel_rates_birth import water_hatching
from heiplanet_models.Pmodel.Pmodel_initial import (
    read_global_settings,
    assemble_filepaths,
    check_all_paths_exist,
    load_all_data,
)

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

FILEPATH_ETL_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings.yaml"

INITIAL_YEAR = 2023
FINAL_YEAR = 2024


def main():
    # Set logger
    logging.basicConfig(level=logging.INFO)

    # Main processor
    for year in range(INITIAL_YEAR, FINAL_YEAR + 1):
        logger.info(f" >>> START Processing year {year} ")

        # 1. Read ETL settings
        ETL_SETTINGS = read_global_settings(
            filepath_configuration_file=FILEPATH_ETL_SETTINGS
        )

        # 2. Assemble paths
        paths = assemble_filepaths(year, **ETL_SETTINGS)  # OK

        # 3. Verify if all the files exist for a given year
        if check_all_paths_exist(path_dict=paths) is False:
            logger.info(f"Year {year} could not be processed.")
            logger.info(f" >>> END Processing year {year} \n")
            continue

        # 4. Load all data
        model_data = load_all_data(paths=paths, etl_settings=ETL_SETTINGS)
        print(model_data)
        print(model_data.rainfall)
        print(model_data.population_density)

        # 5. Calculate water capacity rates
        water_hatching_rate = water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        print(f"Water hatching rate: {water_hatching_rate}")

        # 6. Carrying capacity rates
        carrying_capacity_rate = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        print(f"Carrying capacity rate: {carrying_capacity_rate}")

        logger.info(f" >>> END Processing year {year} \n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
