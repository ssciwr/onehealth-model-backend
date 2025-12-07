import logging
from re import A

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

FILEPATH_TEMPERATURE_DATA = "data/in/Pratik_datalake/dataset_temperature_dummy.nc"
FILEPATH_RAINFALL_DATA = "data/in/Pratik_datalake/dataset_rainfall_dummy.nc"
FILEPATH_POPULATION_DATA = "data/in/Pratik_datalake/dataset_population_dummy.nc"


def main():
    
    # 1. Load Datasets
    temperature_dataset = xr.open_dataset(FILEPATH_TEMPERATURE_DATA)
    logger.info(f"Temperature dataset info: \n{temperature_dataset}")

    rainfall_dataset = xr.open_dataset(FILEPATH_RAINFALL_DATA)
    logger.info(f"Precipitation dataset info: \n{rainfall_dataset}")

    population_dataset = xr.open_dataset(FILEPATH_POPULATION_DATA)
    logger.info(f"Population dataset info: \n{population_dataset}")

    # 2. Verify dimensions
    dimensions_temperature = (3, 2, 4) # (longitude, latitude, time)
    dimensions_rainfall = (3, 2, 4) # (longitude, latitude, time)
    dimensions_population = (3, 2, 1) # (longitude, latitude, time)
    
    

    assert temperature_dataset.temperature.shape == dimensions_temperature, "Temperature dataset dimensions are incorrect"
    #assert rainfall_dataset.dims == ("longitude", "latitude", "time"), "Rainfall dataset dimensions are incorrect"
    #assert population_dataset.dims == ("longitude", "latitude", "time"), "Population dataset dimensions are incorrect"


        # 4. Load all data
        # model_data = load_all_data(paths=paths, etl_settings=ETL_SETTINGS)
        # logger.info(model_data)

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
        level=logging.INFO,
    )
    main()
