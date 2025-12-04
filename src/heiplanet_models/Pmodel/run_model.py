import logging

from heiplanet_models.Pmodel.Pmodel_rates_development import (
    mosq_dev_j,
    mosq_dev_i,
    mosq_dev_e,
    carrying_capacity,
)

from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    water_hatching,
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
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
        logger.info(model_data)

        # --------- Manual verification of rates_birth functions -----------
        # a. mosq_birth
        mosq_birth_rate = mosq_birth(temperature=model_data.temperature)
        logger.info(f"Mosquito birth rate: {mosq_birth_rate}")

        # b. mosq dia hatch
        hatch = mosq_dia_hatch(
            temperature=model_data.temperature_mean, latitude=model_data.latitude
        )
        logger.info(f"Mosquito diapause hatching rate: {hatch}")

        # c. mosq dia lay
        mosq_dia_lay_rate = mosq_dia_lay(
            temperature=model_data.temperature_mean, latitude=model_data.latitude
        )
        logger.info(f"Mosquito diapause laying rate: {mosq_dia_lay_rate}")

        # d. water hatching
        water_hatching_rate = water_hatching(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        logger.info(f"Water hatching rate: {water_hatching_rate}")

        # e. mosq_dev_j
        mosq_dev_j_rate = mosq_dev_j(temperature=model_data.temperature)
        logger.info(f"Mosquito 'j' stage development rate: {mosq_dev_j_rate.values}")

        # f. mosq_dev_i
        mosq_dev_i_rate = mosq_dev_i(temperature=model_data.temperature)
        logger.info(f"Mosquito 'i' stage development rate: {mosq_dev_i_rate.values}")

        # g. mosq_dev_e
        mosq_dev_e_rate = mosq_dev_e(temperature=model_data.temperature)
        logger.info(f"Mosquito 'e' stage development rate: {mosq_dev_e_rate.values}")

        # h. carrying_capacity
        carrying_capacity_rate = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )
        logger.info(f"Carrying capacity rate: {carrying_capacity_rate.values}")

        logger.info(f" >>> END Processing year {year} \n")


if __name__ == "__main__":
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.INFO,
    )
    main()
