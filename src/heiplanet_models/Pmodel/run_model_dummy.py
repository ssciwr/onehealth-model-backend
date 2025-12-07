import logging
import logging
from pathlib import Path

import xarray as xr

from heiplanet_models.Pmodel.Pmodel_ode import call_function
from heiplanet_models.Pmodel.Pmodel_rates_mortality import (
    mosq_mort_e,
    mosq_mort_j,
    mosq_mort_a,
    mosq_surv_ed,
)

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

FILEPATH_ETL_SETTINGS = "./src/heiplanet_models/Pmodel/global_settings_dummy.yaml"

INITIAL_YEAR = 2023
FINAL_YEAR = 2024

def assemble_filepaths_no_year(**etl_settings) -> dict[str, Path]:
    """Assemble file paths for datasets for a given year based on ETL settings.

    Args:
        year (int): The year for which to assemble dataset file paths.
        **etl_settings: Arbitrary keyword arguments containing ETL configuration, must include
            'ingestion' with 'path_root_datasets' and 'filename_components'.

    Returns:
        dict[str, Path]: Dictionary mapping dataset names to their corresponding file paths as Path objects.

    Raises:
        KeyError: If required keys are missing in etl_settings.
        TypeError: If the year is not an integer or settings are malformed.
    """
    # TODO: move to utils.py in the future



    path_root = Path(etl_settings["ingestion"]["path_root_datasets"])
    filename_components = etl_settings["ingestion"]["filename_components"]

    dict_paths = {
        dataset_name: path_root
        / f"{comp['prefix']}{comp['suffix'] or ''}{comp['extension']}"
        for dataset_name, comp in filename_components.items()
    }
    return dict_paths

def print_time_slices(dataarray):
    """
    Print each time slice of a DataArray.
    """
    for i, t in enumerate(dataarray.time):
        print(f"Time slice {i}:")
        print(dataarray.sel(time=t).values)
        print("-" * 40)
        print("\n")

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
        paths = assemble_filepaths_no_year(**ETL_SETTINGS)  # OK


        # 3. Verify if all the files exist for a given year
        if check_all_paths_exist(path_dict=paths) is False:
            logger.info(f"Year {year} could not be processed.")
            logger.info(f" >>> END Processing year {year} \n")
            continue

        # # 4. Load all data
        model_data = load_all_data(paths=paths, etl_settings=ETL_SETTINGS)
        logger.info(model_data)


        # Verify carrying capacity function
        CC = carrying_capacity(
            rainfall_data=model_data.rainfall,
            population_data=model_data.population_density,
        )

        logger.info(f"Dim. Carrying capacity data: {CC.values.shape}")
        logger.info(f"Carrying capacity data: \n{print_time_slices(CC)}")
        logger.info(f"Carrying capacity data: \n{CC[:,:,2].values}")



        # --------- Manual verification of rates_birth functions -----------
        # # a. mosq_birth
        # mosq_birth_rate = mosq_birth(temperature=model_data.temperature)
        # logger.info(f"Mosquito birth rate: {mosq_birth_rate}")

        # # b. mosq dia hatch
        # hatch = mosq_dia_hatch(
        #     temperature=model_data.temperature_mean, latitude=model_data.latitude
        # )
        # logger.info(f"Mosquito diapause hatching rate: {hatch}")

        # # c. mosq dia lay
        # mosq_dia_lay_rate = mosq_dia_lay(
        #     temperature=model_data.temperature_mean, latitude=model_data.latitude
        # )
        # logger.info(f"Mosquito diapause laying rate: {mosq_dia_lay_rate}")

        # # d. water hatching
        # water_hatching_rate = water_hatching(
        #     rainfall_data=model_data.rainfall,
        #     population_data=model_data.population_density,
        # )
        # logger.info(f"Water hatching rate: {water_hatching_rate}")

        # # e. mosq_dev_j
        # mosq_dev_j_rate = mosq_dev_j(temperature=model_data.temperature)
        # logger.info(f"Mosquito 'j' stage development rate: {mosq_dev_j_rate.values}")

        # # f. mosq_dev_i
        # mosq_dev_i_rate = mosq_dev_i(temperature=model_data.temperature)
        # logger.info(f"Mosquito 'i' stage development rate: {mosq_dev_i_rate.values}")

        # # g. mosq_dev_e
        # mosq_dev_e_rate = mosq_dev_e(temperature=model_data.temperature)
        # logger.info(f"Mosquito 'e' stage development rate: {mosq_dev_e_rate.values}")

        # # h. carrying_capacity
        # carrying_capacity_rate = carrying_capacity(
        #     rainfall_data=model_data.rainfall,
        #     population_data=model_data.population_density,
        # )
        # logger.info(f"Carrying capacity rate: {carrying_capacity_rate.values}")

        # # i. mosq_mort_e
        # mosq_mort_e_rate = mosq_mort_e(temperature=model_data.temperature)
        # logger.info(f"Mosquito egg mortality rate: {mosq_mort_e_rate.values}")

        # # j. mosq_mort_j
        # mosq_mort_j_rate = mosq_mort_j(temperature=model_data.temperature)
        # logger.info(f"Mosquito 'j' stage mortality rate: {mosq_mort_j_rate.values}")

        # # k. mosq_mort_a
        # mosq_mort_a_rate = mosq_mort_a(temperature=model_data.temperature_mean)
        # logger.info(f"Mosquito adult mortality rate: {mosq_mort_a_rate.values}")

        # # l. mosq_surv_ed
        # mosq_surv_ed_rate = mosq_surv_ed(
        #     temperature=model_data.temperature,
        #     step_t=ETL_SETTINGS["ode_system"]["time_step"],
        # )
        # logger.info(f"Mosquito survival rate: {mosq_surv_ed_rate.values}")



        # CC  = carrying_capacity(
        #     rainfall_data=model_data.rainfall,
        #     population_data=model_data.population_density,
        # )

        # egg_active = water_hatching(
        #     rainfall_data=model_data.rainfall,
        #     population_data=model_data.population_density,
        # )

        # v = call_function(
        #     v=model_data.initial_conditions,
        #     Temp=model_data.temperature,
        #     Tmean=model_data.temperature_mean,
        #     LAT=model_data.latitude,
        #     CC=CC,
        #     egg_activate=egg_active,
        #     step_t=ETL_SETTINGS["ode_system"]["time_step"],
        # )

        # print(v.shape)

        # compartments = ['egg_non_diapause', 'egg_diapause', 'juvenile', 'immature_adult', 'adult']

        # # Create a dict of DataArrays, one for each compartment
        # data_vars = {}
        # for i, name in enumerate(compartments):
        #     data_vars[name] = xr.DataArray(
        #         v[..., i, :],  # shape: (longitude, latitude, time)
        #         dims=('longitude', 'latitude', 'time'),
        #         coords={
        #             'longitude': model_data.temperature_mean['longitude'],
        #             'latitude': model_data.temperature_mean['latitude'],
        #             'time': model_data.temperature_mean['time'],
        #         },
        #         name=name
        #     )

        # # Combine into a Dataset
        # v_ds = xr.Dataset(data_vars)

        # # Save to NetCDF if desired
        # v_ds.to_netcdf(f'mosquito_population_year_{year}.nc')

        # logger.info(f" >>> END Processing year {year} \n")

        


if __name__ == "__main__":
    logging.basicConfig(
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        level=logging.DEBUG,
    )
    main()
