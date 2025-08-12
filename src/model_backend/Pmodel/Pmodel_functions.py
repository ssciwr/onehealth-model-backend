import logging

import numpy as np
import xarray as xr

from Pmodel_params import (
    CONSTANTS_CARRYING_CAPACITY,
    CONSTANTS_WATER_HATCHING,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def carrying_capacity(
    rainfall_data: xr.DataArray,
    population_data: xr.DataArray,
    constants: dict = CONSTANTS_CARRYING_CAPACITY,
) -> xr.DataArray:
    """
    Compute the carrying capacity over time using rainfall and population density.

    Ref. Equation: # 14
    Name: Juvenile carrying capacity

    Args:
        rainfall_data: Rainfall DataArray with dimensions (time, latitude, longitude).
        population_data: Population density DataArray with dimensions (time, latitude, longitude) or (latitude, longitude).
        constants: Dictionary with keys 'ALPHA_RAIN', 'ALPHA_DENS', 'GAMMA', 'LAMBDA'.

    Returns:
        xr.DataArray: Carrying capacity with the same shape as rainfall_data.
    """

    ALPHA_RAIN = constants["ALPHA_RAIN"]
    ALPHA_DENS = constants["ALPHA_DENS"]
    GAMMA = constants["GAMMA"]
    LAMBDA = constants["LAMBDA"]

    logger.debug(
        f"rainfall_data dims: {rainfall_data.dims}, population_data dims: {population_data.dims}"
    )

    logger.debug(f"Rainfall data: {rainfall_data.values}")

    logger.debug(f"Population data: {population_data.values}")

    # Align population_data to rainfall_data if needed (assume population_data is time-invariant if no time dim)
    if "time" not in population_data.dims:
        population_data = population_data.expand_dims(time=rainfall_data.time)

    # Initialize A with zeros
    A = xr.zeros_like(rainfall_data)

    # Initial condition
    population_data_initial_slice = population_data.sel(time=rainfall_data.time[0])
    A.loc[dict(time=rainfall_data.time[0])] = (
        ALPHA_RAIN * rainfall_data.sel(time=rainfall_data.time[0])
        + ALPHA_DENS * population_data_initial_slice
    )

    # Recursive computation
    for k in range(1, len(rainfall_data.time)):
        prev_rainfall = A.sel(time=rainfall_data.time[k - 1])
        curr_rainfall = rainfall_data.sel(time=rainfall_data.time[k])
        A.loc[dict(time=rainfall_data.time[k])] = (
            GAMMA * prev_rainfall
            + ALPHA_RAIN * curr_rainfall
            + ALPHA_DENS * population_data_initial_slice
        )

    # Apply scaling factor
    for k in range(1, len(rainfall_data.time)):
        factor = (1 - GAMMA) / (1 - GAMMA ** (k + 1))
        A.loc[dict(time=rainfall_data.time[k])] = factor * A.sel(
            time=rainfall_data.time[k]
        )

    # Final scaling
    result = A * LAMBDA
    logger.debug("End of the function")

    return result


def water_hatching(
    rainfall_data: xr.DataArray,
    population_data: xr.DataArray,
    constants: dict = CONSTANTS_WATER_HATCHING,
) -> xr.DataArray:
    """
    Compute the water hatching probability or rate as a function of rainfall and population data.

    Ref. Equation: # 13
    Name: Hatching fraction depending in human density and rainfall

    Args:
        rainfall_data (xr.DataArray): Rainfall data with dimensions (time, latitude, longitude).
        population_data (xr.DataArray): Population density data with dimensions (time, latitude, longitude) or (latitude, longitude).
        constants (dict, optional): Dictionary with keys 'E_OPT', 'E_VAR', 'E_0', 'E_RAT', 'E_DENS', 'E_FAC'.

    Returns:
        xr.DataArray: Combined hatching probability or rate with the same shape as rainfall_data.
    """

    E_OPT = constants["E_OPT"]
    E_VAR = constants["E_VAR"]
    E_0 = constants["E_0"]
    E_RAT = constants["E_RAT"]
    E_DENS = constants["E_DENS"]
    E_FAC = constants["E_FAC"]

    # Handle missing 'time' dimension in population_data
    if "time" not in population_data.dims:
        population_data = population_data.expand_dims(time=rainfall_data.time)

    population_hatch = E_DENS / (E_DENS + np.exp(-E_FAC * population_data))
    logger.debug(population_hatch.values)

    exp_term = np.exp(-E_VAR * (rainfall_data - E_OPT) ** 2)
    rainfall_hatch = (1 + E_0) * exp_term / (exp_term + E_0)

    logger.debug(f"Dimension rainfall_hatch: {rainfall_hatch.shape}")
    logger.debug(f"Dimension rainfall_hatch: {rainfall_hatch.dims}")
    logger.debug(f"Dimension population_hatch: {population_hatch.shape}")
    logger.debug(f"Dimension population_hatch: {population_hatch.dims}")

    try:
        # Use the first time slice for density adjustment and broadcast to match rainfall_hatch
        population_hatch_no_time = population_hatch.isel(time=0).drop_vars("time")
        population_hatch_broadcasted = population_hatch_no_time.expand_dims(
            time=rainfall_hatch.coords["time"]
        )
    except Exception as e:
        raise RuntimeError("Error broadcasting population density adjustment: " + str(e))

    # Weighted combination (element-wise)
    result = ((1 - E_RAT) * rainfall_hatch) + (E_RAT * population_hatch_broadcasted)
    logger.debug(f"Shape result: {result.shape}")

    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    from Pmodel_initial import load_data

    model_data = load_data(time_step=10)
    model_data.print_attributes()

    def print_slices(dataset, value):
        for i in range(value):  # for indices 0, 1, 2
            print(f"Slice at time index {i}:")
            print(dataset.isel(time=i).values)
            print()  # Blank line for readability

    constants_dummy_cc = {
        "ALPHA_RAIN": 10.0,
        "ALPHA_DENS": 10.0,
        "GAMMA": 10.0,
        "LAMBDA": 10.0,
    }

    CC = carrying_capacity(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
        constants=constants_dummy_cc,
    )
    print_slices(CC, 3)

    constants_dummy_hatch = {
        "E_OPT": 8.0,
        "E_VAR": 0.05,
        "E_0": 1.5,
        "E_RAT": 0.2,
        "E_DENS": 0.01,
        "E_FAC": 0.01,
    }

    hatch = water_hatching(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
        constants=constants_dummy_hatch,
    )
    print_slices(hatch, value=3)

    print(model_data.initial_conditions)
