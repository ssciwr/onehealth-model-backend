import logging

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---- General Functions ----


def revolution_angle(days: int) -> float:
    """
    Calculate Earth's revolution angle in radians for a given day of the year.

    Args:
        days (int): Day of the year (1-366).

    Returns:
        float: Revolution angle in radians.
    """
    REVOLUTION_CONST_1 = 0.2163108
    REVOLUTION_CONST_2 = 0.9671396
    REVOLUTION_CONST_3 = 0.0086
    REVOLUTION_CONST_4 = 367
    REVOLUTION_CONST_5 = 186

    theta = REVOLUTION_CONST_1 + 2 * np.arctan(
        REVOLUTION_CONST_2
        * np.tan(REVOLUTION_CONST_3 * ((days % REVOLUTION_CONST_4) - REVOLUTION_CONST_5))
    )
    return theta


def declination_angle(revolution_angle: float) -> float:
    """
    Predict the sun's declination angle in radians.

    Args:
        revolution_angle (float): Revolution angle in radians.

    Returns:
        float: Sun's declination angle in radians.
    """
    DECLINATION_CONST = 0.39795
    phi = np.arcsin(DECLINATION_CONST * np.cos(revolution_angle))
    return phi


def daylight_forsythe(
    latitude: float, declination_angle: float, daylight_coefficient: float
) -> float:
    """
    Calculate daylight hours using Forsythe's method.

    Args:
        latitude (float): Latitude in degrees (-90 to 90).
        declination_angle (float): Declination angle in radians.
        daylight_coefficient (float): Daylight coefficient in degrees.

    Returns:
        float: Daylight hours.

    Raises:
        ValueError: If latitude is not between -90 and 90 degrees.
    """
    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    latitude_rad = np.deg2rad(latitude)
    daylight_coeff_rad = np.deg2rad(daylight_coefficient)

    angle_calculation = (np.sin(daylight_coeff_rad) + np.sin(declination_angle)) / (
        np.cos(latitude_rad) * np.cos(declination_angle)
    )

    daylight = np.real(24 - (24 / np.pi) * np.arccos(angle_calculation))
    return daylight


# ---- Specific Function from the model
def mosq_dia_lay(temperature: xr.DataArray, latitude: xr.DataArray) -> xr.DataArray:
    """
    Calculate mosquito diapause induction based on daylight and latitude.

    Args:
        temperature (xr.DataArray): Temperature data with dimensions (longitude, latitude, time).
        latitude (xr.DataArray): Latitude values (1D array).

    Returns:
        xr.DataArray: Diapause induction array with same dims/coords as input.

    Raises:
        ValueError: If input dimensions are incompatible.
    """
    RATIO_DIA_LAY = 0.5
    DAYS_YEAR = 365
    HALF_DAYS_YEAR = 183

    # Validate input dimensions
    if temperature.ndim != 3:
        raise ValueError("Temperature array must be 3D (longitude, latitude, time).")
    if latitude.ndim != 1:
        raise ValueError("Latitude array must be 1D.")

    n_longitude, n_latitude, n_time = temperature.shape

    # Calculate declination angle for each day
    days = np.arange(1, n_time + 1)
    theta = revolution_angle(days)
    phi = declination_angle(theta)  # shape: (n_time,)

    # Prepare output array
    out = temperature.copy().values

    # Conversion fromn degrees to radians
    latitude_radians = np.deg2rad(latitude.values)

    for k in range(n_latitude):
        lat = latitude_radians[k]
        part_1 = np.sin(lat) * np.sin(phi)
        part_2 = np.cos(lat) * np.cos(phi)

        # daylight calculation
        daylight = 24 - (24 / np.pi) * np.arccos((part_1 / part_2) + 0j).real  # OK
        # Set non-real daylight values to zero (e.g., above Arctic Circle)
        daylight[~np.isreal(daylight)] = 0
        CPP = 10.058 + 0.08965 * lat
        daylight[daylight > CPP] = 0

        # Assign daylight to all longitudes for this latitude
        out[:, k, :] = np.tile(daylight, (n_longitude, 1))

    # No diapause induction in the first half of each year
    n_years = n_time // DAYS_YEAR
    for k in range(n_years):
        start = k * DAYS_YEAR
        end = start + HALF_DAYS_YEAR
        out[:, :, start:end] = 0

    out[out > 0] = 1
    out = out * RATIO_DIA_LAY

    return xr.DataArray(out, dims=temperature.dims, coords=temperature.coords)


def mosq_dia_hatch(temperature: xr.DataArray, latitude: xr.DataArray) -> xr.DataArray:
    """
    Calculate mosquito diapause hatching based on temperature and daylight.

    Args:
        temperature (xr.DataArray): Temperature data with dimensions (longitude, latitude, time).
        latitude (xr.DataArray): Latitude values (1D array).

    Returns:
        xr.DataArray: Diapause hatching array with same dims/coords as input.

    Raises:
        ValueError: If input dimensions are incompatible.
    """

    PERIOD = 7
    CPP = 11.25
    CTT = 11.0
    RATIO_DIA_HATCH = 0.1

    # Validate input dimensions
    if temperature.ndim != 3:
        raise ValueError("Temperature array must be 3D (longitude, latitude, time).")
    if latitude.ndim != 1:
        raise ValueError("Latitude array must be 1D.")

    n_longitude, n_latitude, n_time = temperature.shape

    # Calculate mean temperature of the last 'PERIOD' days and compare to CTT
    out = temperature.copy().values
    for k in range(n_time - 1, PERIOD - 2, -1):
        out[:, :, k] = np.mean(out[:, :, k - PERIOD + 1 : k + 1], axis=2)

    # Mask values below critical temperature threshold
    out[out < CTT] = 0

    # Calculate declination angle for each day
    days = np.arange(1, n_time + 1)
    theta = revolution_angle(days)
    phi = declination_angle(theta)  # shape: (n_time,)

    # Conversion fromn degrees to radians
    latitude_radians = np.deg2rad(latitude.values)

    for k in range(n_latitude):
        lat = latitude_radians[k]  # radians

        # daylight for all days

        part_1 = np.sin(lat) * np.sin(phi).squeeze()  # OK
        part_2 = np.cos(lat) * np.cos(phi).squeeze()  # OK

        # daylight calculation
        daylight = 24 - (24 / np.pi) * np.arccos((part_1 / part_2) + 0j).real  # OK

        # Replicate daylight shape to (x, z)
        daylight_matrix = np.tile(daylight, (n_longitude, 1))
        # Mask where daylight < CPP
        T_help = out[:, k, :]
        T_help[daylight_matrix < CPP] = 0
        out[:, k, :] = T_help

    # Set NaNs to 0
    out = np.nan_to_num(out, nan=0.0)

    # Binarize and scale
    out[out > 0] = RATIO_DIA_HATCH

    # Return as xarray.DataArray with same dims/coords as input
    return xr.DataArray(out, dims=temperature.dims, coords=temperature.coords)


def mosq_surv_ed(temperature: np.ndarray, step_t: float | None = None) -> np.ndarray:
    """
    Calculate mosquito survival rate based on temperature using a rolling minimum.

    Args:
        temperature (np.ndarray): 3D array of temperature values with shape (longitude, latitude, time).
        step_t (float | None, optional): Time step in days. If provided, the first 90 * step_t time steps are removed.

    Returns:
        np.ndarray: Survival rate array with same shape as input, or with reduced time dimension if step_t is provided.

    Notes:
        - Applies a rolling minimum along the time axis.
        - Uses a survival formula based on temperature.
        - If step_t is specified, removes the first 90 * step_t time steps.
    """
    ED_SURV_BL = 1.0

    n_longitude, n_latitude, n_time = temperature.shape

    # Rolling minimum along the time axis (axis=2)
    out = temperature.copy()

    for k in range(1, n_time):
        out[:, :, k] = np.minimum(out[:, :, k - 1], out[:, :, k])

    # Remove the first 90*step_t time steps if step_t is provided
    if step_t is not None:
        remove_steps = int(90 * step_t)
        out = out[:, :, remove_steps:]

    # Apply the survival formula
    out = ED_SURV_BL * 0.93 * np.exp(-0.5 * ((out - 11.68) / 15.67) ** 6)

    return out


# ---- MAIN FUNCTION
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # or INFO in production
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting program...")
