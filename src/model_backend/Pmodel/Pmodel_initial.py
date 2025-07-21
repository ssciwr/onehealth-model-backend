from pathlib import Path
from dataclasses import dataclass

import xarray as xr
import numpy as np


PATH_DATASET_TEMPERATURE = Path(
    "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
)
PATH_DATASET_RAINFALL = Path(
    "data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"
)
PATH_DATASET_POPULATION = Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc")

K1 = 625
K2 = 100
VARIABLES = ["eggs", "ed", "juv", "imm", "adults"]


@dataclass
class PmodelInitial:
    def __init__(self):
        pass


def load_dataset(
    path_dataset, variable_name=None, dimension_order=None, decode_times=True, **kwargs
):
    """
    Generic function to load data from a NetCDF dataset

    Args:
        path_dataset: Path to the dataset file
        variable_name: Name of the variable to extract (if None, returns the full dataset)
        dimension_order: Optional tuple of dimensions to transpose the dataset to
        decode_times: Whether to decode times in the NetCDF file
        **kwargs: Additional arguments to pass to xr.open_dataset

    Returns:
        Either the full dataset or a specific variable
    """
    # Open the dataset (lazily)
    dataset = xr.open_dataset(
        filename_or_obj=path_dataset, decode_times=decode_times, **kwargs
    )

    # Transpose dimensions if specified (also lazy)
    if dimension_order is not None:
        dataset = dataset.transpose(*dimension_order)

    # Return either the whole dataset or just the requested variable
    if variable_name is not None:
        return dataset[variable_name]

    return dataset


def load_human_population_density(
    path_population_dataset,
    variable_name="total-population",
):
    return load_dataset(
        path_dataset=path_population_dataset,
        variable_name=variable_name,
        decode_times=False,
        dimension_order=("lon", "lat", "time"),
    )


def load_initial(filepath_previous, sizes):
    """
    Load initial conditions for the model from a NetCDF file or create default values

    Args:
        filepath_previous: Path to previous simulation output NetCDF file
        sizes: Tuple containing dimensions (size_latitude, size_longitude)

    Returns:
        v0: numpy array with dimensions (size_latitude, size_longitude, num_variables)
    """

    # Assumming the order is ["longitude", "latitude", "model_variable"]
    size_longitude = sizes[0]
    size_latitude = sizes[1]

    # Five differential equations in the model

    number_variables = len(VARIABLES)

    # Create an initial matrix full of zeros
    # Ref. Dimensions [latitude, longitude, model_variable]
    v0 = np.zeros((size_latitude, size_longitude, number_variables), dtype=np.float64)

    if Path(filepath_previous).exists():
        # Open NetCDF file containing initial conditions
        initial_conditions = load_dataset(filepath_previous)

        # Extract the last time step for each variable
        for i, var in enumerate(VARIABLES):
            v0[:, :, i] = initial_conditions[var].isel(time=-1).values8
    else:
        # If no previous file exists, initialize eggs_dia (ed) with 625*100
        v0[:, :, 1] = K1 * K2 * np.ones((size_latitude, size_longitude))

    return v0


def load_latitude(path_latitude_dataset, variable_name):
    # From ERA5 Land
    return load_dataset(
        path_dataset=path_latitude_dataset,
        variable_name=variable_name,
    )


def load_rainfall(path_rainfall_dataset, variable_name):
    return load_dataset(
        path_dataset=path_rainfall_dataset,
        variable_name=variable_name,
        dimension_order=("longitude", "latitude", "time"),
    )


def load_temperature(path_temperature_dataset, variable_name, time_step):
    # Load and transpose the dataset
    var_Temperature_mean = load_dataset(
        path_dataset=path_temperature_dataset,
        variable_name=variable_name,
        dimension_order=("longitude", "latitude", "valid_time"),
    )

    size_longitudes, size_latitudes, size_time = var_Temperature_mean.shape

    var_Temperature = np.zeros(
        (size_longitudes, size_latitudes, size_time * time_step), dtype=np.float64
    )

    return (var_Temperature, var_Temperature_mean)


def load_data() -> PmodelInitial:
    pass


if __name__ == "__main__":

    # Load Population Density
    print("----------- Density ------------")
    dens = load_human_population_density(
        path_population_dataset=PATH_DATASET_POPULATION, variable_name="dens"
    )
    print(dens.shape)

    # Load Temperature Dataset
    print("----------- Temperature ------------")
    T, Tmean = load_temperature(
        path_temperature_dataset=PATH_DATASET_TEMPERATURE,
        variable_name="t2m",
        time_step=10,
    )
    print(Tmean.shape)
    print(T.shape)

    # Load Rainfall Dataset
    print("----------- Rainfall ------------")
    rainfall = load_rainfall(
        path_rainfall_dataset=PATH_DATASET_RAINFALL, variable_name="tp"
    )
    print(rainfall.shape)

    # Load Latitude Dataset
    print("----------- Latitudes ------------")
    latitudes = load_latitude(
        path_latitude_dataset=PATH_DATASET_TEMPERATURE, variable_name="latitude"
    )
    print(latitudes.shape)

    # Load initial
    print("----------- Initial Conditions ------------")
    sizes = Tmean.shape
    v0 = load_initial(filepath_previous="previous", sizes=(sizes[0], sizes[1]))
    print(v0.shape)
