"""Module for loading and managing initial conditions and input data for the Pratik model.

This module provides functions and data structures for loading, processing, and
organizing the input data required by the Pratik epidemiological model, including:
- Temperature and rainfall data from ERA5Land
- Population density data
- Initial conditions for model variables
- Utility functions for data loading and preparation

The main entry point is the load_data function, which returns a PmodelInitial object
containing all necessary data for the model.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Union, Tuple, Optional

import xarray as xr
import numpy as np


PATH_DATASET_TEMPERATURE = Path(
    "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
)
# PATH_DATASET_RAINFALL = Path(
#    "data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"
# )
# PATH_DATASET_POPULATION = Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc")

PATH_DATASET_RAINFALL = Path("data/in/Pratik_datalake/pr_dummy.nc")
PATH_DATASET_POPULATION = Path("data/in/Pratik_datalake/dense_dummy.nc")


K1 = 625
K2 = 100
VARIABLES = ["eggs", "ed", "juv", "imm", "adults"]

CHUNKING_SCHEME = {"longitude": 90, "latitude": 45, "time": 10}
COORDINATES = ("longitude", "latitude", "time")


@dataclass
class PmodelInitial:
    temperature: xr.DataArray
    temperature_mean: xr.DataArray
    rainfall: xr.DataArray
    latitude: xr.DataArray
    population_density: xr.DataArray
    initial_conditions: np.ndarray = None

    def get_dimensions(self) -> tuple[int, int]:
        """Returns the longitude and latitude dimensions of the temperature data.

        Returns:
            A tuple containing (longitude_size, latitude_size).
        """
        return self.temperature_mean.shape[:2]

    def print_attributes(self) -> None:
        """Prints all attributes of the class instance.

        This method dynamically lists all attributes in the instance,
        making it useful when the class is updated with new attributes.
        """
        print("PmodelInitial Attributes:")

        # Get all instance variables using __annotations__ to include type hints
        attributes = self.__annotations__ if hasattr(self, "__annotations__") else {}

        # Add any instance attributes that might not be in annotations
        for attr_name in dir(self):
            # Filter out methods, private attributes, and special methods
            if (
                not attr_name.startswith("_")
                and not callable(getattr(self, attr_name))
                and attr_name not in attributes
            ):
                attributes[attr_name] = type(getattr(self, attr_name)).__name__

        # Print each attribute with its type and value information
        for attr_name, attr_type in attributes.items():
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    print(f"  - {attr_name}: None")
                elif hasattr(attr_value, "shape"):
                    print(
                        f"  - {attr_name}: {type(attr_value).__name__} with shape {attr_value.shape}"
                    )
                else:
                    print(f"  - {attr_name}: {type(attr_value).__name__}")
            else:
                print(f"  - {attr_name}: Not set")


def load_dataset(
    path_dataset: Path | str,
    variable_name: str = None,
    dimension_order: tuple = None,
    decode_times: bool = True,
    chunks: Optional[dict] = None,
    new_names: Optional[dict] = None,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Generic function to load data from a NetCDF dataset in chunks.

    Args:
        path_dataset: Path to the dataset file.
        variable_name: Name of the variable to extract. If None, returns the full dataset.
        dimension_order: Optional tuple of dimensions to transpose the dataset to.
        decode_times: Whether to decode times in the NetCDF file.
        chunks: Optional dict for Dask chunk sizes (e.g., {'time': 10, 'latitude': 100, 'longitude': 100}).
        **kwargs: Additional arguments to pass to xr.open_dataset.

    Returns:
        Either the full dataset (xr.Dataset) or a specific variable (xr.DataArray).
    """
    dataset = xr.open_dataset(
        filename_or_obj=path_dataset,
        decode_times=decode_times,
        **kwargs,
    )

    dataset = dataset.rename(new_names)
    dataset = dataset.chunk(chunks)

    if dimension_order is not None:
        dataset = dataset.transpose(*dimension_order)

    if variable_name is not None:
        return dataset[variable_name]

    return dataset


def load_human_population_density(
    path_population_dataset: Path | str,
    variable_name: str = "total-population",
    chunks: Optional[dict] = None,
) -> xr.DataArray:
    """Loads population density data from a NetCDF file.

    Args:
        path_population_dataset: Path to the population density dataset file.
        variable_name: Name of the variable to extract.

    Returns:
        A DataArray containing the population density data.
    """
    rainfall_dataset = load_dataset(
        path_dataset=path_population_dataset,
        variable_name=variable_name,
        decode_times=True,
        dimension_order=("longitude", "latitude", "time"),
        chunks=chunks,
        new_names={"lon": "longitude", "lat": "latitude", "time": "time"},
    )

    return rainfall_dataset


def load_initial_conditions(
    filepath_previous: Path | str, sizes: tuple[int, int]
) -> np.ndarray:
    """Load initial conditions for the model from a NetCDF file or create default values.

    Args:
        filepath_previous: Path to previous simulation output NetCDF file.
        sizes: Tuple containing dimensions (size_longitude, size_latitude).

    Returns:
        v0: numpy array with dimensions (size_latitude, size_longitude, num_variables).
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
            v0[:, :, i] = initial_conditions[var].isel(time=-1).values
    else:
        # If no previous file exists, initialize eggs_dia (ed) with 625*100
        v0[:, :, 1] = K1 * K2 * np.ones((size_latitude, size_longitude))

    return v0


def load_latitude(path_latitude_dataset: Path | str, variable_name: str) -> xr.DataArray:
    """Loads latitude data from a NetCDF file (typically from ERA5 Land).

    Args:
        path_latitude_dataset: Path to the dataset containing latitude data.
        variable_name: Name of the latitude variable in the dataset.

    Returns:
        A DataArray containing the latitude data.
    """
    return load_dataset(
        path_dataset=path_latitude_dataset,
        variable_name=variable_name,
        chunks=CHUNKING_SCHEME,
        new_names={"valid_time": "time"},
    )


def load_rainfall(
    path_rainfall_dataset: Path | str, variable_name: str, chunks: Optional[dict] = None
) -> xr.DataArray:
    """Loads rainfall data from a NetCDF file and transposes to standard dimension order.

    Args:
        path_rainfall_dataset: Path to the rainfall dataset file.
        variable_name: Name of the rainfall variable in the dataset.

    Returns:
        A DataArray containing the rainfall data with dimensions (longitude, latitude, time).
    """
    return load_dataset(
        path_dataset=path_rainfall_dataset,
        variable_name=variable_name,
        dimension_order=("longitude", "latitude", "time"),
        chunks=chunks,
    )


def load_temperature(
    path_temperature_dataset: Path | str,
    variable_name: str,
    time_step: int,
    chunks: Optional[dict] = None,
) -> tuple[np.ndarray, xr.DataArray]:
    """Loads temperature data from a NetCDF file and prepares arrays for processing.

    Args:
        path_temperature_dataset: Path to the temperature dataset file.
        variable_name: Name of the temperature variable in the dataset.
        time_step: Time step parameter for expanding the temperature array.

    Returns:
        A tuple containing:
            - var_Temperature: A numpy array for storing processed temperature data
              with dimensions (longitude, latitude, time*time_step).
            - var_Temperature_mean: The original temperature data as a DataArray
              with dimensions (longitude, latitude, time).
    """
    # Load and transpose the dataset
    var_Temperature_mean = load_dataset(
        path_dataset=path_temperature_dataset,
        variable_name=variable_name,
        dimension_order=("longitude", "latitude", "time"),
        new_names={"valid_time": "time"},
        chunks=CHUNKING_SCHEME,
    )

    size_longitudes, size_latitudes, size_time = var_Temperature_mean.shape

    var_Temperature = np.zeros(
        (size_longitudes, size_latitudes, size_time * time_step), dtype=np.float64
    )

    return (var_Temperature, var_Temperature_mean)


def load_data(
    path_temperature: Path | str = PATH_DATASET_TEMPERATURE,
    path_rainfall: Path | str = PATH_DATASET_RAINFALL,
    path_population: Path | str = PATH_DATASET_POPULATION,
    filepath_previous: Path | str = None,
    time_step: int = 1,
) -> PmodelInitial:
    """Load all required data for the model into a PmodelInitial object.

    This function serves as the main entry point for loading all model data,
    including environmental and demographic inputs, as well as optional
    initial conditions from a previous simulation.

    Args:
        path_temperature: Path to temperature dataset.
        path_rainfall: Path to rainfall dataset.
        path_population: Path to population density dataset.
        filepath_previous: Path to previous simulation results (if any).
        time_step: Time step parameter for temperature data processing.

    Returns:
        A PmodelInitial object containing all required data.
    """

    # Load temperature data
    var_temperature, var_temperature_mean = load_temperature(
        path_temperature_dataset=path_temperature,
        variable_name="t2m",
        time_step=time_step,
        chunks=CHUNKING_SCHEME,
    )

    # Load rainfall data
    var_rainfall = load_rainfall(
        path_rainfall_dataset=path_rainfall,
        variable_name="tp",
        chunks=CHUNKING_SCHEME,
    )

    # Load latitude data
    var_latitude = load_latitude(
        path_latitude_dataset=path_temperature, variable_name="latitude"
    )

    # Load population density data
    var_population = load_human_population_density(
        path_population_dataset=path_population,
        variable_name="dens",
        chunks=CHUNKING_SCHEME,
    )

    # TODO: remove for production code.
    var_population_aligned = var_population.interp(
        longitude=var_rainfall.longitude,
        latitude=var_rainfall.latitude,
        method="linear",
    )
    # var_population_aligned, var_rainfall = xr.align(
    #     var_population,
    #     var_rainfall,
    #     join="outer",  # or "inner", "outer", "left", "right" depending on your use case
    # )

    # Create model data container
    model_data = PmodelInitial(
        temperature_mean=var_temperature_mean,
        temperature=var_temperature,
        rainfall=var_rainfall,
        latitude=var_latitude,
        population_density=var_population_aligned,
    )

    # Load initial conditions if provided
    if filepath_previous:
        sizes = model_data.get_dimensions()
        model_data.initial_conditions = load_initial_conditions(
            filepath_previous=filepath_previous, sizes=sizes
        )

    return model_data


if __name__ == "__main__":

    # Load lazily without chunks

    # dataset = load_dataset(
    #     path_dataset=PATH_DATASET_RAINFALL,
    #     variable_name="tp",
    #     dimension_order=("longitude", "latitude", "time"),
    #     decode_times=True,
    #     chunks=CHUNKING_SCHEME,
    # )

    # dens = dataset

    # print(dens)
    # print("\nDask chunking info:")
    # print(dens.data)

    model_data = load_data(
        time_step=10,
        # Uncomment to test with initial conditions
        # filepath_previous="previous"
    )

    # Print all attributes of the model data object
    print("\n----------- Model Attributes ------------")
    model_data.print_attributes()

    # Print information about loaded data
    print("\n----------- Data Shapes ------------")
    print(f"Temperature:\t\t{model_data.temperature.shape}")
    print(f"Temperature mean: \t{model_data.temperature_mean.shape}")
    print(f"Rainfall:\t\t{model_data.rainfall.shape}")
    print(f"Latitude:\t\t{model_data.latitude.shape}")
    print(f"Population Density:\t{model_data.population_density.shape}")

    # Print additional information if initial conditions were loaded
    if model_data.initial_conditions is not None:
        print(f"Initial Conditions: {model_data.initial_conditions.shape}")
    else:
        print("No initial conditions loaded")

    print("\n----------- Model Dimensions ------------")
    print(f"Model dimensions (longitude, latitude): {model_data.get_dimensions()}")

    print("\nData loading complete.")

    print(model_data.population_density.values.shape)
    print(model_data.rainfall.values.shape)
