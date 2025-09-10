import logging

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MOSQUITO_J,
    CONSTANTS_MOSQUITO_I,
    CONSTANTS_MOSQUITO_E,
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

    CONST_1 = CONSTANTS_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_J["CONST_4"]

    # # New function briere with coeffiecint with initial data collection, for Sandra and Zia model
    # # Commented on purpose
    # temperature = q*temperatute*(temperatute - T0 )*((Tm - temperatute)**(1/2));

    T_out = CONST_1 - CONST_2 * temperature + CONST_3 * temperature**2
    T_out = CONST_4 / T_out
    return T_out


def mosq_dev_i(temperature: np.ndarray) -> np.ndarray:
    """Calculate mosquito development rate for the 'i' stage as a function of temperature.

    This function implements the Octave mosq_dev_i formula for the 'i' developmental
    stage of mosquitoes, using a temperature-dependent mathematical model.

    Args:
        temperature (numpy.ndarray): Array of temperature values in degrees Celsius.

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature value.
    """

    CONST_1 = CONSTANTS_MOSQUITO_I["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_I["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_I["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_I["CONST_4"]

    # # New function briere with coeffiecint with initial data collection, for Sandra and Zia model
    # # Commented on purpose
    # temperature = q*temperature*(temperature - T0 )*((Tm - temperature)**(1/2));

    T_out = CONST_1 - CONST_2 * temperature + CONST_3 * temperature**2
    T_out = CONST_4 / T_out
    return T_out


def mosq_dev_e(temperature: np.ndarray) -> np.ndarray:
    """
    Calculates the mosquito egg development rate using the Brière model.

    Args:
        temperature (numpy.ndarray): Array of temperature values in degrees Celsius.

    Returns:
        numpy.ndarray: Array of development rates, elementwise for each temperature value.

    Raises:
        ValueError: If input is not a numpy ndarray.
    """
    q = CONSTANTS_MOSQUITO_E["q"]
    T0 = CONSTANTS_MOSQUITO_E["T0"]
    Tm = CONSTANTS_MOSQUITO_E["Tm"]

    # # New function briere with coeffiecint with initial data collection, for Sandra and Zia model
    T_out = (
        q * temperature * (temperature - T0) * ((Tm - temperature) ** (1 / 2))
    )  # Corrected ^ to **

    # Found in original code
    # T_out = CONST_1 - CONST_2 * temperatute + CONST_3 * temperatute**2;
    # T_out = CONST_4 ./ T_OUT;

    return T_out


def carrying_capacity(
    rainfall_data: xr.DataArray, population_data: xr.DataArray
) -> xr.DataArray:
    """Computes the carrying capacity over time using rainfall and population density.

    This function calculates the juvenile carrying capacity (Ref. Equation #14)
    based on a recursive formula involving historical rainfall and population
    density. It is optimized to use vectorized operations for performance.

    Args:
        rainfall_data: A DataArray containing rainfall data with dimensions
            (time, latitude, longitude).
        population_data: A DataArray containing population density data. It can
            be time-dependent with dimensions (time, latitude, longitude) or
            time-independent with dimensions (latitude, longitude).

    Returns:
        An xarray DataArray representing the carrying capacity, with the same
        shape as the `rainfall_data`.
    """
    ALPHA_RAIN = CONSTANTS_CARRYING_CAPACITY["ALPHA_RAIN"]
    ALPHA_DENS = CONSTANTS_CARRYING_CAPACITY["ALPHA_DENS"]
    GAMMA = CONSTANTS_CARRYING_CAPACITY["GAMMA"]
    LAMBDA = CONSTANTS_CARRYING_CAPACITY["LAMBDA"]

    logger.debug(f"rainfall_data dims: {rainfall_data.dims}, shape: {rainfall_data.shape}")
    logger.debug(f"population_data dims: {population_data.dims}, shape: {population_data.shape}")

    if "time" not in population_data.dims:
        population_data = population_data.expand_dims(time=rainfall_data.time)

    population_data_initial_slice = population_data.isel(time=0)

    # Create an array of powers of GAMMA
    time_size = len(rainfall_data.time)
    gamma_powers = xr.DataArray(
        GAMMA ** np.arange(time_size - 1, -1, -1), dims=["time"]
    )

    # Calculate the term related to rainfall and population density
    rainfall_term = ALPHA_RAIN * rainfall_data + ALPHA_DENS * population_data_initial_slice
    
    # Apply the recursive formula using cumulative sum
    # This is equivalent to the loop but vectorized
    A = (rainfall_term * gamma_powers).cumsum(dim="time", skipna=True) / gamma_powers

    # Calculate the scaling factor
    k = xr.DataArray(np.arange(time_size), dims="time")
    scaling_factor = (1 - GAMMA) / (1 - GAMMA ** (k + 1))
    
    # Apply scaling factors
    A_scaled = A * scaling_factor

    # Final scaling
    result = A_scaled * LAMBDA
    logger.debug("End of the function")

    return result


if __name__ == "__main__":

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    from heiplanet_models.Pmodel.Pmodel_initial import load_data

    def print_slices(dataset, value):

        for i in range(value):  # for indices 0, 1, 2
            print(f"Slice at time index {i}:")
            print(dataset.isel(time=i).values)
            print()  # Blank line for readability

    model_data = load_data(time_step=10)
    print("---- Model attributes ")
    model_data.print_attributes()

    print("\n---- function: mosq_dev_j()")
    temperatute = np.array([[15.0, 20.0], [25.0, 30.0]])
    print(temperatute)
    print(mosq_dev_j(temperatute))

    print("\n---- function: mosq_dev_i()")
    print(mosq_dev_i(temperatute))

    print("\n---- function: mosq_dev_e()")
    print(mosq_dev_e(temperatute))

    print("\n---- function: carrying_capacity()")
    result = carrying_capacity(
        rainfall_data=model_data.rainfall, population_data=model_data.population_density
    )
    print(f"{print_slices(result, 3)}")
