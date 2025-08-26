"""Initial data loading and preprocessing utilities for the OneHealth model backend.

This module provides functions to efficiently load, preprocess, and align large
geospatial datasets (temperature, rainfall, population) for use in the OneHealth
model. It supports chunked reading via Dask, robust error handling, and logging.

Typical usage example:
    model_input = load_data()
"""

import logging
from pathlib import Path
from typing import Any

import xarray as xr
import numpy as np


from src.heiplanet_models.Pmodel.config import (
    CHUNKING_SCHEME,
    COORDINATES_ORDER,
    MODEL_VARIABLES,
    PATH_DATASETS_SANDBOX,
    TIME_STEP,
)
from src.heiplanet_models.Pmodel.Pmodel_input import PmodelInput

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def load_dataset(
    path_dataset: Path | str,
    decode_times: bool = True,
    names_dimensions: dict[str, str] | None = None,
    chunks: dict[str, int] | None = None,
    dimension_order: tuple[str, ...] | None = None,
    variable_name: str | None = None,
    **kwargs: Any,
) -> xr.Dataset | xr.DataArray:
    """Load an xarray dataset with optional renaming, chunking, and variable extraction.

    Args:
        path_dataset: Path to the dataset file.
        decode_times: Whether to decode time coordinates.
        names_dimensions: Optional mapping for renaming dataset dimensions.
        chunks: Optional Dask chunking scheme for efficient reading.
        dimension_order: Optional tuple specifying desired dimension order.
        variable_name: Optional variable name to extract from the dataset.
        **kwargs: Additional keyword arguments for xarray.open_dataset.

    Returns:
        xr.Dataset or xr.DataArray: Loaded dataset or extracted variable.

    Raises:
        Exception: If dataset loading, renaming, transposing, or variable extraction fails.
    """
    try:
        dataset = xr.open_dataset(
            filename_or_obj=path_dataset,
            chunks=chunks,
            decode_times=decode_times,
            **kwargs,
        )
    except Exception as e:
        logger.error(f"Failed to open dataset '{path_dataset}': {e}")
        raise

    if names_dimensions:
        try:
            dataset = dataset.rename(names_dimensions)
        except Exception as e:
            logger.error(
                f"Failed to rename dimensions in dataset '{path_dataset}': {e}"
            )
            raise

    if dimension_order:
        try:
            dataset = dataset.transpose(*dimension_order)
        except Exception as e:
            logger.error(f"Failed to transpose dataset '{path_dataset}': {e}")
            raise

    if variable_name:
        if variable_name not in dataset:
            logger.error(
                f"Variable '{variable_name}' not found in dataset '{path_dataset}'."
            )
            raise KeyError(f"Variable '{variable_name}' not found.")
        return dataset[variable_name]
    return dataset


def load_initial_conditions(
    filepath_previous: Path | str | None = None,
    sizes: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Load or initialize the model state variables.

    Args:
        filepath_previous: Optional path to previous initial conditions file.
        sizes: Tuple of (longitude, latitude) grid sizes.

    Returns:
        np.ndarray: Initialized or loaded state variable array.

    Raises:
        Exception: If loading or extracting previous conditions fails.
        KeyError: If a required variable is missing in previous conditions.
    """
    K1 = 625
    K2 = 100
    n_longitude, n_latitude = sizes
    n_vars = len(MODEL_VARIABLES)
    v0 = np.zeros((n_longitude, n_latitude, n_vars), dtype=np.float64)

    if filepath_previous is None or not Path(filepath_previous).exists():
        v0[:, :, 1] = K1 * K2
        logger.info("Initialized initial conditions with default values.")
    else:
        try:
            ds = load_dataset(filepath_previous)
        except Exception as e:
            logger.error(
                f"Failed to load previous initial conditions from '{filepath_previous}': {e}"
            )
            raise
        for i, var in enumerate(MODEL_VARIABLES):
            if var not in ds:
                logger.error(
                    f"Variable '{var}' not found in previous conditions dataset."
                )
                raise KeyError(
                    f"Variable '{var}' not found in previous conditions dataset."
                )
            try:
                v0[:, :, i] = ds[var].isel(time=-1).values
            except Exception as e:
                logger.error(
                    f"Failed to extract variable '{var}' from previous conditions: {e}"
                )
                raise
        logger.info("Loaded initial conditions from previous file.")
    return v0


def load_temperature(
    path_dataset: Path | str,
    time_step: int = 1,
    **kwargs: Any,
) -> tuple[np.ndarray, xr.DataArray]:
    """Load temperature and expand time resolution by repeating along the time axis.

    Args:
        path_dataset: Path to the temperature dataset file.
        time_step: Number of times to repeat each time slice.
        **kwargs: Additional keyword arguments for dataset loading.

    Returns:
        tuple[np.ndarray, xr.DataArray]: Expanded temperature array and mean temperature DataArray.

    Raises:
        Exception: If loading or expanding the temperature array fails.
    """
    try:
        var_temperature_mean: xr.DataArray = load_dataset(
            path_dataset=path_dataset, **kwargs
        )
    except Exception as e:
        logger.error(f"Failed to load temperature dataset '{path_dataset}': {e}")
        raise

    try:
        var_temperature = np.repeat(var_temperature_mean.values, time_step, axis=2)
    except Exception as e:
        logger.error(f"Failed to expand temperature array: {e}")
        raise

    return var_temperature, var_temperature_mean


def align_xarray_datasets(
    misaligned_dataset: xr.DataArray,
    fixed_dataset: xr.DataArray,
) -> xr.DataArray:
    """Align coordinates of one DataArray to another using interpolation.

    Args:
        misaligned_dataset: DataArray to be interpolated (e.g., population density).
        fixed_dataset: DataArray providing target longitude and latitude coordinates (e.g., rainfall).

    Returns:
        xr.DataArray: Interpolated DataArray aligned to the target grid.

    Raises:
        Exception: If interpolation fails.
    """
    try:
        return misaligned_dataset.interp(
            longitude=fixed_dataset.longitude,
            latitude=fixed_dataset.latitude,
            method="linear",
        )
    except Exception as e:
        logger.error("Failed to align coordinates using interpolation: %s", e)
        raise


def load_data(
    path_temperature: Path | str = PATH_DATASETS_SANDBOX["TEMPERATURE"],
    path_rainfall: Path | str = PATH_DATASETS_SANDBOX["RAINFALL"],
    path_population: Path | str = PATH_DATASETS_SANDBOX["HUMAN_POPULATION"],
    filepath_previous: Path | str | None = None,
    time_step: int = TIME_STEP,
) -> PmodelInput:
    """Load all input datasets into a unified model input structure.

    Args:
        path_temperature: Path to temperature dataset.
        path_rainfall: Path to rainfall dataset.
        path_population: Path to population dataset.
        filepath_previous: Optional path to previous initial conditions.
        time_step: Number of times to repeat each time slice for temperature.

    Returns:
        PmodelInput: Structured model input data.

    Raises:
        Exception: If any dataset loading or alignment fails.
    """
    try:
        var_latitude = load_dataset(
            path_dataset=path_temperature,
            names_dimensions={"time": "time"},
            chunks=CHUNKING_SCHEME,
            variable_name="latitude",
        )
        var_population = load_dataset(
            path_dataset=path_population,
            decode_times=True,
            names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
            chunks=CHUNKING_SCHEME,
            dimension_order=COORDINATES_ORDER,
            variable_name="dens",
        )
        var_rainfall = load_dataset(
            path_dataset=path_rainfall,
            chunks=CHUNKING_SCHEME,
            dimension_order=COORDINATES_ORDER,
            variable_name="tp",
        )
        var_temperature, var_temperature_mean = load_temperature(
            path_dataset=path_temperature,
            names_dimensions={"time": "time"},
            chunks=CHUNKING_SCHEME,
            dimension_order=COORDINATES_ORDER,
            variable_name="t2m",
            time_step=time_step,
        )
        sizes = var_temperature_mean.shape[:2]
        initial_conditions = load_initial_conditions(
            filepath_previous=filepath_previous,
            sizes=sizes,
        )
        var_population_aligned = align_xarray_datasets(var_population, var_rainfall)
    except Exception as e:
        logger.error(f"Failed to load model input data: {e}")
        raise

    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=var_latitude,
        population_density=var_population_aligned,
        rainfall=var_rainfall,
        temperature=var_temperature,
        temperature_mean=var_temperature_mean,
    )


def main() -> None:
    """Main entry point for module testing and demonstration.

    Loads rainfall and other datasets, prints their attributes and shapes,
    and demonstrates the data loading pipeline.
    """
    logging.basicConfig(level=logging.INFO)
    try:
        dataset = load_dataset(
            path_dataset=PATH_DATASETS_SANDBOX["RAINFALL"],
            decode_times=True,
            chunks=CHUNKING_SCHEME,
            dimension_order=("longitude", "latitude", "time"),
            variable_name="tp",
        )
        logger.info("Loaded rainfall dataset successfully.")
        logger.info(f"Dask chunking info: {dataset.data}")

        model_data = load_data(
            time_step=10,
            # Uncomment to test with initial conditions
            # filepath_previous="previous"
        )
        logger.info("Model data loaded successfully.")

        logger.info("----------- Model Attributes ------------")
        model_data.print_attributes()

        logger.info("----------- Data Shapes ------------")
        logger.info(f"Initial Conditions:\t{model_data.initial_conditions.shape}")
        logger.info(f"Latitude:\t\t{model_data.latitude.shape}")
        logger.info(f"Population Density:\t{model_data.population_density.shape}")
        logger.info(f"Rainfall:\t\t{model_data.rainfall.shape}")
        logger.info(f"Temperature:\t\t{model_data.temperature.shape}")
        logger.info(f"Temperature mean: \t{model_data.temperature_mean.shape}")

        logger.info("Data loading complete.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()
