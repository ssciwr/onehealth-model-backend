import logging

import numpy as np

from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MORTALITY_MOSQUITO_E

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
