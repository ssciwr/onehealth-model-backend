import logging

import numpy as np

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MORTALITY_MOSQUITO_E,
    CONSTANTS_MORTALITY_MOSQUITO_J,
    CONSTANTS_MORTALITY_MOSQUITO_A,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_mort_e(temperature: np.ndarray) -> np.ndarray:
    """Calculates the daily mortality rate for eggs.

    This function calculates the daily mortality rate for eggs based on temperature,
    following the implementation from the Octave version of the model.

    Args:
        temperature (np.ndarray): A numpy.ndarray containing temperature values.

    Returns:
        numpy.ndarray: A numpy.ndarray of the same shape as `temperature` with the calculated
        daily mortality rates for eggs.
    """

    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_5"]

    T_out = CONST_1 * np.exp(CONST_2 * ((temperature - CONST_3) / CONST_4) ** CONST_5)
    T_out = -np.log(T_out)

    return T_out


def mosq_mort_j(temperature: np.ndarray) -> np.ndarray:
    """Calculates the daily mortality rate for juvenile mosquitoes.

    This function calculates the daily mortality rate for juvenile mosquitoes based on temperature,
    following the implementation from the Octave version of the model.

    Args:
        temperature (numpy.ndarray): A numpy.ndarray containing temperature values.

    Returns:
        numpy.ndarray: A numpy.ndarray of the same shape as `temperature` with the calculated
        daily mortality rates for juvenile mosquitoes.
    """

    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_5"]

    T_out = CONST_1 * np.exp(CONST_2 * ((temperature - CONST_3) / CONST_4) ** CONST_5)

    # To avoid numerical issues when logarithm will be calculated.
    T_out = np.where(T_out > 0, T_out, 1e-12)

    T_out = -np.log(T_out)

    return T_out


def mosq_mort_a(temperature: np.ndarray) -> np.ndarray:
    """
    Calculates the daily mortality rate for adult mosquitoes based on temperature.

    This function implements the Octave `mosq_mort_a` logic in Python.

    Args:
        temperature (np.ndarray): Array of temperature values.

    Returns:
        numpy.ndarray: Array of daily adult mosquito mortality rates, elementwise for each temperature value.
    """
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_5"]
    CONST_6 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_6"]

    T_out: np.ndarray = np.array(temperature).copy()
    mask_pos = T_out > 0
    mask_zero = ~mask_pos

    # For T > 0
    T_out[mask_pos] = (
        CONST_1
        * np.exp(CONST_2 * ((T_out[mask_pos] - CONST_3) / CONST_4) ** CONST_5)
        * T_out[mask_pos] ** CONST_6
    )
    # For T <= 0
    T_out[mask_zero] = CONST_1 * np.exp(
        CONST_2 * ((T_out[mask_zero] - CONST_3) / CONST_4) ** CONST_5
    )

    T_out = -np.log(T_out)
    return T_out
