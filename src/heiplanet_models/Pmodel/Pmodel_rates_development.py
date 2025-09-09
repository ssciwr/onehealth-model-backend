import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MOSQUITO_J,
    CONSTANTS_CARRYING_CAPACITY,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_dev_j(temperature: np.ndarray) -> np.ndarray:
    """Calculate mosquito juvenile development rate as a function of temperature.

    This function implements the Octave mosq_dev_j formula for mosquito juvenile
    development rate, using a temperature-dependent mathematical model.

    Args:
        temperature (numpy.ndarray): Array of temperature values (°C).

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature.
    """
    # TODO: Status --> it works

    CONST_1 = CONSTANTS_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_J["CONST_4"]

    # #  New function briere with coeffiecient with initial data collection
    # #  Commented on purpose, this is documented in original Octave/Matlab code
    # Tout = q*T*(T - T0 )*((Tm - T)^(1/2));

    T_out = CONST_1 - CONST_2 * temperature + CONST_3 * temperature**2
    T_out = CONST_4 / T_out
    return T_out


def mosq_dev_i(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_dev_i function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with development rates applied elementwise
    """
    # q = 1.695638e-04;
    # T0 = 3.750303e+00;
    # Tm = 3.553575e+01;

    #  new function briere with coffiecint with initial data collection, for Sandra and Zia model
    # T = q*T*(T - T0 )*((Tm - T)**(1/2));

    # Old parameter description in original model
    T_out = 50.1 - 3.574 * T + 0.069 * T**2
    T_out = 1.0 / T_out
    return T_out


def mosq_dev_e(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_dev_e function (Brière model).

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with development rates applied elementwise
    """
    q = 0.0001246068
    T0 = -7.0024634748
    Tm = 34.1519214674

    # Apply the Brière model formula elementwise
    T_out = q * T * (T - T0) * np.sqrt(np.maximum(Tm - T, 0))

    # Found in original code
    # T_out = 50.1 - 3.574 * T + 0.069 * T**2;
    # T = 1 ./ T;

    # Ensure no negative values due to sqrt of negative numbers
    T_out = np.where((T > T0) & (T < Tm), T_out, 0.0)
    return T_out


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


if __name__ == "__main__":

    print("\n---- function: mosq_dev_j()")
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    print(T)
    # Output:
    # [[15. 20.]
    #  [25. 30.]]

    print(mosq_dev_j(T))
