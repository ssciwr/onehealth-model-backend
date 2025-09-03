import logging
from typing import Union

import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_REVOLUTION_ANGLE
from heiplanet_models.Pmodel.Pmodel_params import CONSTANT_DECLINATION_ANGLE
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_BIRTH
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_DIAPAUSE_LAY
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING
from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_WATER_HATCHING

# from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_WATER_HATCHING


MIN_LAT_DEGRESS = -90
MAX_LAT_DEGREES = 90
HOURS_PER_DAY = 24

DAYS_YEAR = 365
HALF_DAYS_YEAR = 183

# ---- Logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---- Helper Functions
def validate_spatial_alignment(arr1: xr.DataArray, arr2: xr.DataArray) -> None:
    """Validates that two xarray DataArrays have aligned spatial coordinates.

    Args:
        arr1 (xr.DataArray): The first DataArray.
        arr2 (xr.DataArray): The second DataArray.

    Raises:
        ValueError: If the 'latitude' or 'longitude' coordinates do not match
                    or if the coordinates are missing.
    """
    try:
        if not np.array_equal(
            arr1.latitude.values, arr2.latitude.values
        ) or not np.array_equal(arr1.longitude.values, arr2.longitude.values):
            raise ValueError(
                "Spatial coordinates ('latitude', 'longitude') of input arrays "
                "must be aligned."
            )
    except AttributeError:
        raise ValueError(
            "Input DataArrays must have 'latitude' and 'longitude' coordinates."
        )


# ---- General functions
def revolution_angle(days: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
    """Calculate Earth's revolution angle in radians for given days of the year.

    This function can process both a single integer day or a numpy array of days.

    Args:
        days (Union[int, np.ndarray]): Day or days of the year (values must be
            in the range 1-366, inclusive).

    Returns:
        Union[float, np.ndarray]: Revolution angle or angles in radians.
            Returns a float for a single day input, and a numpy array for an
            array input.

    Raises:
        TypeError: If `days` is not an integer or an array of integers.
        ValueError: If any day is not in the range 1 to 366.
    """

    CONST_1 = CONSTANTS_REVOLUTION_ANGLE["CONST_1"]
    CONST_2 = CONSTANTS_REVOLUTION_ANGLE["CONST_2"]
    CONST_3 = CONSTANTS_REVOLUTION_ANGLE["CONST_3"]
    CONST_4 = CONSTANTS_REVOLUTION_ANGLE["CONST_4"]
    CONST_5 = CONSTANTS_REVOLUTION_ANGLE["CONST_5"]

    days_arr = np.asarray(days)

    if not np.issubdtype(days_arr.dtype, np.integer):
        raise TypeError("Input 'days' must be an integer or an array of integers.")

    if not np.all((days_arr >= 1) & (days_arr <= 366)):
        raise ValueError("All 'days' must be in the range 1 to 366.")

    theta = CONST_1 + 2 * np.arctan(
        CONST_2 * np.tan(CONST_3 * ((days_arr % CONST_4) - CONST_5))
    )

    return theta.item() if days_arr.ndim == 0 else theta


def declination_angle(revolution_angle: float) -> float:
    """Predict the sun's declination angle in radians.

    Args:
        revolution_angle (float): Revolution angle in radians.

    Returns:
        float: Sun's declination angle in radians.
    """

    phi = np.arcsin(CONSTANT_DECLINATION_ANGLE * np.cos(revolution_angle))
    return phi


def daylight_forsythe(
    latitude: float, declination_angle: float, daylight_coefficient: float
) -> float:
    """Calculate daylight hours using Forsythe's method.

    Args:
        latitude (float): Latitude in degrees (-90 to 90).
        declination_angle (float): Declination angle in radians.
        daylight_coefficient (float): Daylight coefficient in degrees.

    Returns:
        float: Daylight hours.

    Raises:
        ValueError: If latitude is not between -90 and 90 degrees.
    """

    if not MIN_LAT_DEGRESS <= latitude <= MAX_LAT_DEGREES:
        raise ValueError("Latitude must be between -90 and 90 degrees.")

    latitude_rad = np.deg2rad(latitude)
    daylight_coeff_rad = np.deg2rad(daylight_coefficient).squeeze()

    angle_calculation = (
        np.sin(daylight_coeff_rad) + np.sin(latitude_rad) * np.sin(declination_angle)
    ) / (np.cos(latitude_rad) * np.cos(declination_angle))

    daylight = np.real(
        HOURS_PER_DAY - (HOURS_PER_DAY / np.pi) * np.arccos(angle_calculation + 0j)
    )
    return daylight


# --- Specific functions
def mosq_birth(temperature: np.ndarray) -> np.ndarray:
    """Calculates the mosquito birth rate based on temperature.

    This function computes the birth rate using a formula that is applied
    only when the temperature is below a certain threshold (CONST_1).
    If the temperature is at or above this threshold, the birth rate is
    considered to be zero.

    Args:
        temperature (np.ndarray): An array of temperature values.

    Returns:
        np.ndarray: An array of the same shape as the input, containing
            the calculated mosquito birth rates.
    """
    CONST_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_BIRTH["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_BIRTH["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_BIRTH["CONST_4"]
    CONST_5 = CONSTANTS_MOSQUITO_BIRTH["CONST_5"]
    CONST_6 = CONSTANTS_MOSQUITO_BIRTH["CONST_6"]

    out = temperature.copy()
    mask = out < CONST_1
    # Only apply the formula where T < CONST_1
    out[mask] = (
        CONST_2
        * np.exp(CONST_3 * ((out[mask] - CONST_4) / CONST_5) ** 2)
        * (CONST_1 - out[mask]) ** CONST_6
    )
    # Set to zero where T >= CONST_1
    out[~mask] = 0
    return out


def mosq_dia_hatch(temperature: xr.DataArray, latitude: xr.DataArray) -> xr.DataArray:
    """Calculates mosquito diapause hatching based on temperature and daylight.

    Hatching is predicted when the rolling average temperature is above a
    critical threshold (`CTT`) and the daily photoperiod is longer than a
    critical length (`CPP`).

    Args:
        temperature (xr.DataArray): 3D DataArray of temperature values
            (longitude, latitude, time).
        latitude (xr.DataArray): 1D DataArray of latitude values.

    Returns:
        xr.DataArray: A 3D DataArray indicating where hatching occurs. Values
            are a constant ratio where conditions are met, and 0 otherwise.

    Raises:
        ValueError: If input arrays have incompatible dimensions.
    """

    PERIOD = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["PERIOD"]
    CPP = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CPP"]
    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    RATIO_DIA_HATCH = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["RATIO_DIA_HATCH"]
    DAYLENGTH_COEFFICIENT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING[
        "DAYLENGTH_COEFFICIENT"
    ]

    # Validate input dimensions
    if temperature.ndim != 3:
        raise ValueError("Temperature array must be 3D (longitude, latitude, time).")
    if latitude.ndim != 1:
        raise ValueError("Latitude array must be 1D.")

    n_longitude, n_latitude, n_time = temperature.shape

    # Calculate mean temperature of the last 'PERIOD' days and compare to CTT
    out = temperature.copy().values
    for k in range(n_time - 1, PERIOD - 2, -1):
        out[:, :, k] = np.mean(out[:, :, (k - PERIOD + 1) : (k + 1)], axis=2)

    # Mask values below critical temperature threshold
    out[out < CTT] = 0

    days = np.arange(1, n_time + 1)
    theta = revolution_angle(days)
    phi = declination_angle(theta)  # shape: (n_time,)

    latitude_degrees = latitude.values

    for k in range(n_latitude):
        lat = latitude_degrees[k]

        daylight = daylight_forsythe(
            latitude=lat,
            declination_angle=phi,
            daylight_coefficient=DAYLENGTH_COEFFICIENT,
        )

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

    RATIO_DIA_LAY = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["RATIO_DIA_LAY"]
    CONST_1 = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["CONST_2"]
    DAYLENGTH_COEFFICIENT = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["DAYLENGTH_COEFFICIENT"]

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
    latitude_degrees = latitude.values

    for k in range(n_latitude):
        lat = latitude_degrees[k]

        daylight = daylight_forsythe(
            latitude=lat,
            declination_angle=phi,
            daylight_coefficient=DAYLENGTH_COEFFICIENT,
        )
        # Set non-real daylight values to zero (e.g., above Arctic Circle)
        daylight[~np.isreal(daylight)] = 0
        CPP = CONST_1 + CONST_2 * lat
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


def water_hatching(
    rainfall_data: xr.DataArray,
    population_data: xr.DataArray,
) -> xr.DataArray:
    """Compute the water hatching probability or rate as a function of rainfall and population data.

    This function calculates a hatching rate based on two factors: a rainfall-dependent term that
    peaks at an optimal rainfall level, and a population-density-dependent term modeling artificial
    breeding sites. The two factors are combined in a weighted average.

    Ref. Equation: # 13
    Name: Hatching fraction depending in human density and rainfall

    Args:
        rainfall_data (xr.DataArray): A 3D DataArray of rainfall data with
            dimensions ('time', 'latitude', 'longitude').
        population_data (xr.DataArray): A DataArray of population density.
            Can be 3D ('time', 'latitude', 'longitude') or 2D
            ('latitude', 'longitude').

    Returns:
        xr.DataArray: A DataArray of the combined hatching probability, with
            the same shape as `rainfall_data`.
    """

    E_OPT = CONSTANTS_WATER_HATCHING["E_OPT"]
    E_VAR = CONSTANTS_WATER_HATCHING["E_VAR"]
    E_0 = CONSTANTS_WATER_HATCHING["E_0"]
    E_RAT = CONSTANTS_WATER_HATCHING["E_RAT"]
    E_DENS = CONSTANTS_WATER_HATCHING["E_DENS"]
    E_FAC = CONSTANTS_WATER_HATCHING["E_FAC"]

    # Validate input alignment first
    validate_spatial_alignment(rainfall_data, population_data)

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
        raise RuntimeError(
            "Error broadcasting population density adjustment: " + str(e)
        )

    # Weighted combination (element-wise)
    result = ((1 - E_RAT) * rainfall_hatch) + (E_RAT * population_hatch_broadcasted)
    logger.debug(f"Shape result: {result.shape}")

    return result


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    from heiplanet_models.Pmodel.Pmodel_initial import load_data

    def print_slices(dataset, value):

        for i in range(value):  # for indices 0, 1, 2
            print(f"Slice at time index {i}:")
            print(dataset.isel(time=i).values)
            print()  # Blank line for readability

    model_data = load_data(time_step=10)
    print("---- Model attributes ")
    model_data.print_attributes()

    print("\n---- function: revolution_angle()")
    print(f"\tRevolution angle, day 1: {revolution_angle(1)}")
    print(f"\tRevolution angle, day 365: {revolution_angle(365)}")
    print(f"\tRevolution angle, day 366: {revolution_angle(366)}")

    print("\n---- function: declination_angle()")
    print(f"\tDeclination angle, np.pi/2: {declination_angle(np.pi / 2)}")
    print(f"\tDeclination angle, -0.409: {declination_angle(-0.409)}")

    print("\n---- function: daylight_forsythe()")
    result = daylight_forsythe(
        latitude=89.999, declination_angle=-0.409, daylight_coefficient=0.0
    )
    print(f"\tDaylight Forsythe: {result}")

    print("\n---- function: mosquito_birth()")
    # TODO: test this function

    print("\n---- function: mosquito_diapause_hatch()")
    result = mosq_dia_hatch(model_data.temperature_mean, model_data.latitude)
    print(f"{print_slices(result, 3)}")

    print("\n---- function: mosquito_diapause_lay()")
    # TODO: test this function

    # print("\n---- function: water_hatching()")
    # hatch = water_hatching(
    #     rainfall_data=model_data.rainfall,
    #     population_data=model_data.population_density,
    # )

    # print_slices(hatch, value=3)
