import logging

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp


def assemble_pmodel():
    pass


def run_pmodel():
    pass


def revolution_angle(days):
    """

    Args:
        J: days of the year

    Return
        theta: revolution angle in radians
    """

    CONST_1 = 0.2163108
    CONST_2 = 0.9671396
    CONST_3 = 0.0086
    CONST_4 = 367
    CONST_5 = 186

    theta = CONST_1 + 2 * np.arctan(
        CONST_2 * np.tan(CONST_3 * ((days % CONST_4) - CONST_5))
    )

    # TODO: Remove theta function with hardcoded constants after testing.
    # theta = 0.2163108 + 2 * np.arctan(
    #    0.9671396 * np.tan(0.0086 * ((days % 367) - 186))
    # )

    return theta


def declination_angle(revolution_angle):
    """Predict the sun's declination angle.

    Args:
        revolution_angle (float): revolution angle in radians

    Returns:
        phi(float): sun's declination angle in radians.
    """
    CONST_1 = 0.39795

    # TODO: Remove phi function with hardcoded constants after testing.
    phi = np.arcsin(CONST_1 * np.cos(revolution_angle))

    return phi


def dayligth_forsythe(latitude, declination_angle, dayligth_coefficient):
    """_summary_

    Args:
        latitude (_type_): _description_
        declination_angle (_type_): _description_
        dayligth_coefficient (_type_): _description_
    """

    dayligth_coefficient = 0

    # Conversion from degrees to radians
    dayligth_coefficient = np.deg2rad(dayligth_coefficient)
    latitude = np.deg2rad(latitude)

    angle_calculation = (np.sin(dayligth_coefficient) + np.sin(declination_angle)) / (
        np.cos(latitude) * np.cos(declination_angle)
    )

    daylight = np.real(24 - (24 / np.pi) * np.arccos(angle_calculation))

    return daylight


def mosq_dia_lay(T: xr.DataArray, LATU: xr.DataArray, step_t):
    """
    Python translation of the Octave mosq_dia_lay function.
    T: xarray.DataArray (time, lat, ...)
    LATU: xarray.DataArray (lat,)
    step_t: int (not used in original code)
    """
    ratio_dia_lay = 0.5

    # Assume T dims: (time, lat, ...)
    x, y, z = T.shape

    # Calculate Phi for each day (z axis is time)
    days = np.arange(1, z + 1).reshape(-1, z)
    Phi = np.arcsin(
        0.39795
        * np.cos(
            0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * ((days % 367) - 186)))
        )
    )

    # Prepare output array
    T_out = T.copy().values

    print(f"Size PHI: {Phi.shape}")
    print(f"Size LATU: {LATU.shape}")

    LATU = np.array([-80, -10, 40, 80], dtype=np.float64)
    latitude_radians = np.deg2rad(LATU)

    for k in range(y):
        lat = latitude_radians[k]
        # print(np.degrees(lat))
        # print(np.sin(lat))
        print("\n")
        print(np.sin(lat) * np.sin(Phi))
        print(np.cos(lat) * np.cos(Phi))

        part_1 = np.sin(lat) * np.sin(Phi)
        part_2 = np.cos(lat) * np.cos(Phi)
        part_3 = 24 - (24 / np.pi) * np.acos((part_1 / part_2) + 0j).real

        print(part_3)
        # daylight calculation
        # daylight = np.real(24 - (24 / np.pi) * np.arccos(part_1))
    #     CPP = 10.058 + 0.08965 * lat
    #     daylight[daylight > CPP] = 0
    #     # Assign daylight to all x for this latitude
    #     T_out[:, k, :] = np.tile(daylight, (x, 1))

    # # No diapause induction in the first half of the year
    # n_years = z // 365
    # for k in range(n_years):
    #     start = k * 365
    #     end = start + 183
    #     T_out[:, :, start:end] = 0

    # T_out[T_out > 0] = 1
    # T_out = T_out * ratio_dia_lay

    # Return as xarray.DataArray with same dims/coords as input
    # return xr.DataArray(T_out, dims=T.dims, coords=T.coords)
    return None


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

    diapause_hatch = mosq_dia_lay(
        T=model_data.temperature_mean, LATU=model_data.latitude, step_t=10
    )

    # print(diapause_hatch.shape)
