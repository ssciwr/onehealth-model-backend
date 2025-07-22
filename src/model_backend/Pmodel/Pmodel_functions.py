from pathlib import Path

import numpy as np
import xarray as xr

from Pmodel_params import (
    ALPHA_DENS,
    ALPHA_RAIN,
    GAMMA,
    LAMBDA,
)


def load_dataset(
    path_dataset: Path | str,
    variable_name: str = None,
    dimension_order: tuple = None,
    decode_times: bool = True,
    **kwargs,
) -> xr.Dataset | xr.DataArray:
    """Generic function to load data from a NetCDF dataset.

    Args:
        path_dataset: Path to the dataset file.
        variable_name: Name of the variable to extract. If None, returns the full dataset.
        dimension_order: Optional tuple of dimensions to transpose the dataset to.
        decode_times: Whether to decode times in the NetCDF file.
        **kwargs: Additional arguments to pass to xr.open_dataset.

    Returns:
        Either the full dataset (xr.Dataset) or a specific variable (xr.DataArray).
    """
    # Open the dataset (lazily)
    dataset = xr.open_dataset(
        filename_or_obj=path_dataset,
        engine="netcdf4",
        **kwargs,
    )

    # Transpose dimensions if specified (also lazy)
    if dimension_order is not None:
        dataset = dataset.transpose(*dimension_order)

    # Return either the whole dataset or just the requested variable
    if variable_name is not None:
        return dataset[variable_name]

    return dataset


def load_rainfall(
    path_rainfall_dataset: Path | str, variable_name: str
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
    )


def load_human_population_density(
    path_population_dataset: Path | str,
    variable_name: str = "total-population",
) -> xr.DataArray:
    """Loads population density data from a NetCDF file.

    Args:
        path_population_dataset: Path to the population density dataset file.
        variable_name: Name of the variable to extract.

    Returns:
        A DataArray containing the population density data.
    """
    return load_dataset(
        path_dataset=path_population_dataset,
        variable_name=variable_name,
        dimension_order=("lon", "lat", "time"),
    )


def capacity(pr: xr.DataArray, dens: xr.DataArray) -> xr.DataArray:
    """
    Vectorized implementation of the Octave `capacity` function using NumPy + xarray.
    """
    # Constants
    ALPHA = 0.001
    BETA = 0.00001
    GAMMA = 0.9
    LAMBDA = 1e6 * 625 * 100

    # Make sure population density has same coordinates as rainfall
    # Also ensure it's squeezed to remove singleton dimensions when possible
    dens = dens.reindex_like(pr.isel(time=0, drop=True)).squeeze(drop=True)

    # Make a copy of the input data to avoid modifying in place
    pr_new = pr.copy(deep=True)

    # Convert to numpy arrays for faster operations
    pr_values = pr_new.values
    dens_values = dens.values

    # Check if dens_values has a singleton dimension and reshape if needed
    if dens_values.ndim == 3 and dens_values.shape[2] == 1:
        dens_values = dens_values.squeeze(axis=2)

    # Apply first step: t=0
    pr_values[..., 0] = ALPHA * pr_values[..., 0] + BETA * dens_values

    # Vectorize the recurrence with numpy arrays
    time_dim = pr_values.shape[-1]
    gamma_powers = GAMMA ** np.arange(time_dim - 1)

    # Precompute contributions for all time steps
    alpha_pr = ALPHA * pr_values[..., 1:]
    beta_dens = BETA * dens_values[..., np.newaxis]  # Broadcast dens along time

    # Weighted cumulative sum over time
    a_cum = np.zeros_like(pr_values)
    a_cum[..., 0] = pr_values[..., 0]

    # Compute weighted sum efficiently
    for t in range(1, time_dim):
        a_cum[..., t] = (
            gamma_powers[t - 1] * a_cum[..., t - 1]
            + alpha_pr[..., t - 1]
            + beta_dens[..., 0]
        )

    # Scaling
    time_indices = np.arange(1, time_dim + 1)
    scale_factors = (1 - GAMMA) / (1 - GAMMA**time_indices)

    # Apply scaling
    pr_scaled = a_cum * scale_factors

    # Final scaling
    pr_final = pr_scaled * LAMBDA

    # Return as xarray DataArray with same coordinates
    return xr.DataArray(pr_final, dims=pr.dims, coords=pr.coords, attrs=pr.attrs)


if __name__ == "__main__":
    # Approach 1. Using container for variables
    from Pmodel_initial import load_data

    model_data = load_data(time_step=10)

    model_data.print_attributes()
    print(type(model_data.rainfall))
    print(capacity(model_data.rainfall, model_data.population_density).shape)

    # Approach 2. Using pure functions
    # PATH_DATASET_TEMPERATURE = Path(
    #     "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
    # )
    # PATH_DATASET_RAINFALL = Path(
    #     "data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"
    # )
    # PATH_DATASET_POPULATION = Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc")

    # pr = load_rainfall(path_rainfall_dataset=PATH_DATASET_RAINFALL, variable_name="tp")
    # dens = load_human_population_density(
    #     path_population_dataset=PATH_DATASET_POPULATION, variable_name="dens"
    # )

    # capacity = capacity(pr=pr, dens=dens)

    # print(capacity.shape)
