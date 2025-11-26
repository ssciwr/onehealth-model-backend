import logging
from typing import Optional

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANTS_MORTALITY_MOSQUITO_E,
    CONSTANTS_MORTALITY_MOSQUITO_J,
    CONSTANTS_MORTALITY_MOSQUITO_A,
    CONSTANTS_MORTALITY_MOSQUITO_ED,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def mosq_mort_e(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for eggs as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_5"]

    T_out = CONST_1 * np.exp(
        CONST_2 * ((temperature.data - CONST_3) / CONST_4) ** CONST_5
    )
    T_out = -np.log(T_out)

    return xr.DataArray(
        T_out, coords=temperature.coords, dims=temperature.dims, name="mosq_mort_e"
    )


def mosq_mort_j(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for juvenile mosquitoes as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_5"]

    T_out = CONST_1 * np.exp(
        CONST_2 * ((temperature.data - CONST_3) / CONST_4) ** CONST_5
    )
    T_out = np.where(T_out > 0, T_out, 1e-12)
    T_out = -np.log(T_out)

    return xr.DataArray(
        T_out, coords=temperature.coords, dims=temperature.dims, name="mosq_mort_j"
    )


def mosq_mort_a(temperature: xr.DataArray) -> xr.DataArray:
    """Calculates the daily mortality rate for adult mosquitoes as an xarray.DataArray."""
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_5"]
    CONST_6 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_6"]

    T = temperature.data.copy()
    mask_pos = T > 0
    mask_zero = ~mask_pos

    T_out = np.empty_like(T)
    # For T > 0
    T_out[mask_pos] = (
        CONST_1
        * np.exp(CONST_2 * ((T[mask_pos] - CONST_3) / CONST_4) ** CONST_5)
        * T[mask_pos] ** CONST_6
    )
    # For T <= 0
    T_out[mask_zero] = CONST_1 * np.exp(
        CONST_2 * ((T[mask_zero] - CONST_3) / CONST_4) ** CONST_5
    )

    T_out = -np.log(T_out)
    return xr.DataArray(
        T_out, coords=temperature.coords, dims=temperature.dims, name="mosq_mort_a"
    )


def mosq_surv_ed(
    temperature: xr.DataArray, step_t: Optional[int] = None
) -> xr.DataArray:
    """
    Calculates mosquito survival rate as a function of temperature, following the Octave mosq_surv_ed function.

    Args:
        temperature (xr.DataArray): 3D DataArray of temperature values (e.g., (x, y, t)).
        step_t (int, optional): Time step, used for winter mortality calculation. Defaults to None.

    Returns:
        xr.DataArray: DataArray of mosquito survival rates with the same shape as `temperature`.
    """
    if not isinstance(temperature, xr.DataArray):
        raise ValueError("Input 'temperature' must be an xarray.DataArray.")
    if temperature.ndim != 3:
        raise ValueError("Input 'temperature' must be a 3D DataArray.")

    ED_SURV_BL = CONSTANTS_MORTALITY_MOSQUITO_ED["ED_SURV_BL"]
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_5"]

    T_out = temperature.data.copy()
    n_time = T_out.shape[-1]

    for k in range(1, n_time):
        T_out[..., k] = np.minimum(T_out[..., k - 1], T_out[..., k])

    # Apply the survival formula
    T_out = (
        ED_SURV_BL
        * CONST_1
        * np.exp(CONST_2 * ((T_out - CONST_3) / CONST_4) ** CONST_5)
    )

    return xr.DataArray(
        T_out, coords=temperature.coords, dims=temperature.dims, name="mosq_surv_ed"
    )
