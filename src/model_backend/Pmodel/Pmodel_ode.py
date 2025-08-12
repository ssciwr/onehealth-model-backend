import logging

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp

from Pmodel_functions import carrying_capacity, water_hatching


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


def mosq_dia_lay(T: xr.DataArray, LATU: xr.DataArray):
    """
    Python translation of the Octave mosq_dia_lay function.
    T: xarray.DataArray (time, lat, ...)
    LATU: xarray.DataArray (lat,)


    """
    # TODO: Status --> Works!

    RATIO_DIA_LAY = 0.5

    # Assume T dims: (time, lat, ...)
    x, y, z = T.shape

    # Calculate Phi for each day (z axis is time)
    days = np.arange(1, z + 1)
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

    # TODO: remove hardcoded vector, this is just for experimenting with the function
    # LATU = np.array([-80, -10, 40, 80], dtype=np.float64)
    # latitude_radians = np.deg2rad(LATU)

    # Conversion fromn degrees to radians
    latitude_radians = np.deg2rad(LATU.values)

    for k in range(y):
        lat = latitude_radians[k]
        print(f"shape phi: {Phi.shape}")
        # print(np.degrees(lat))
        # print(np.sin(lat))

        part_1 = np.sin(lat) * np.sin(Phi).squeeze()  # OK
        part_2 = np.cos(lat) * np.cos(Phi).squeeze()  # OK

        # daylight calculation
        daylight = 24 - (24 / np.pi) * np.arccos((part_1 / part_2) + 0j).real  # OK
        # Set non-real daylight values to zero (e.g., above Arctic Circle)
        daylight[~np.isreal(daylight)] = 0
        # print(daylight)

        CPP = 10.058 + 0.08965 * lat
        daylight[daylight > CPP] = 0
        # Assign daylight to all x for this latitude
        T_out[:, k, :] = np.tile(daylight, (x, 1))

    # # No diapause induction in the first half of the year
    n_years = z // 365
    for k in range(n_years):
        start = k * 365
        end = start + 183
        T_out[:, :, start:end] = 0

    T_out[T_out > 0] = 1
    T_out = T_out * RATIO_DIA_LAY

    # Return as xarray.DataArray with same dims/coords as input
    return xr.DataArray(T_out, dims=T.dims, coords=T.coords)


def mosq_dia_hatch(T: xr.DataArray, LATU: xr.DataArray, step_t=None):
    """
    Python translation of the Octave mosq_dia_hatch function.
    T: xarray.DataArray (time, lat, ...)
    LATU: xarray.DataArray (lat,)
    step_t: unused, for interface compatibility
    """

    # TODO: Status --> it works!
    # TODO: duplicated code in comparison to mosq_dia_lay (daylight calculation)
    # TODO: improve with vectorized implementation

    PERIOD = 7
    CPP = 11.25
    CTT = 11.0
    RATIO_DIA_HATCH = 0.1

    # Assume T dims: (time, lat, ...)
    x, y, z = T.shape

    # Calculate mean temperature of the last 'period' days and compare to CTT
    T_out = T.copy().values
    for k in range(z - 1, PERIOD - 2, -1):  # z-1 down to period-1 (Python 0-based)
        T_out[:, :, k] = np.mean(T_out[:, :, k - PERIOD + 1 : k + 1], axis=2)

    T_out[T_out < CTT] = 0

    # Calculate day length and compare to CPP
    days = np.arange(1, z + 1)
    Phi = np.arcsin(
        0.39795
        * np.cos(
            0.2163108 + 2 * np.arctan(0.9671396 * np.tan(0.0086 * ((days % 367) - 186)))
        )
    )

    # TODO: remove hardcoded vector, this is just for experimenting with the function
    # LATU = np.array([-80, -10, 40, 80], dtype=np.float64)
    # latitude_radians = np.deg2rad(LATU)

    # Conversion fromn degrees to radians
    latitude_radians = np.deg2rad(LATU.values)

    for k in range(y):
        lat = latitude_radians[k]  # radians

        # daylight for all days

        part_1 = np.sin(lat) * np.sin(Phi).squeeze()  # OK
        part_2 = np.cos(lat) * np.cos(Phi).squeeze()  # OK

        # daylight calculation
        daylight = 24 - (24 / np.pi) * np.arccos((part_1 / part_2) + 0j).real  # OK

        # Replicate daylight shape to (x, z)
        daylight_matrix = np.tile(daylight, (x, 1))
        # Mask where daylight < CPP
        T_help = T_out[:, k, :]
        T_help[daylight_matrix < CPP] = 0
        T_out[:, k, :] = T_help

    # Set NaNs to 0
    T_out = np.nan_to_num(T_out, nan=0.0)

    # Binarize and scale
    T_out[T_out > 0] = RATIO_DIA_HATCH

    # Return as xarray.DataArray with same dims/coords as input
    return xr.DataArray(T_out, dims=T.dims, coords=T.coords)


def mosq_surv_ed(T: np.ndarray, step_t=None):
    """
    Python translation of the Octave mosq_surv_ed function.
    T: numpy.ndarray (time, lat, ...)
    step_t: time step (optional, used for winter mortality calculation)
    """
    # TODO: Status --> it works
    # TODO: refactor to vectorized implementation

    ED_SURV_BL = 1.0

    # Rolling minimum along the time axis (axis=2)
    T_out = T.copy()
    x, y, z = T_out.shape
    for k in range(1, z):
        T_out[:, :, k] = np.minimum(T_out[:, :, k - 1], T_out[:, :, k])

    # Uncomment the following lines if you want to remove the first 90*step_t time steps
    # if step_t is not None:
    #     remove_steps = int(90 * step_t)
    #     T_out = T_out[:, :, remove_steps:]

    # Apply the survival formula
    T_out = ED_SURV_BL * 0.93 * np.exp(-0.5 * ((T_out - 11.68) / 15.67) ** 6)

    # Return as xarray.DataArray with same dims/coords as input (adjust if time steps removed)
    return T_out


def mosq_birth(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_birth function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with birth rates applied elementwise
    """
    T_out = T.copy()
    mask = T_out < 38.8
    # Only apply the formula where T < 38.8
    T_out[mask] = (
        33.2
        * np.exp(-0.5 * ((T_out[mask] - 70.3) / 14.1) ** 2)
        * (38.8 - T_out[mask]) ** 1.5
    )
    # Set to zero where T >= 38.8
    T_out[~mask] = 0
    return T_out


def mosq_dev_j(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_dev_j function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with development rates applied elementwise
    """
    # TODO: Status --> it works

    # Old parameter description in original model
    T_out = 82.42 - 4.87 * T + 0.08 * T**2
    T_out = 1.0 / T_out
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


def for_cycle_equation(v, Temp, Tmean, LAT, CC, egg_activate, step_t):
    diapause_lay = mosq_dia_lay(Tmean, LAT)
    diapause_hatch = mosq_dia_hatch(Tmean, LAT)
    ed_survival = mosq_surv_ed(Temp, step_t)

    shape_output = (
        v.shape[0],
        v.shape[1],
        5,
        int(Temp.shape[2] / step_t),
    )
    v_out = np.zeros(shape=shape_output)

    for t in range(Temp.shape[2]):
        T = Temp[:, :, t]
        birth = mosq_birth(T)
        dev_j = mosq_dev_j(T)
        dev_i = mosq_dev_i(T)
        dev_e = 1.0 / 7.1  # original model

        # Octave: ceil(t/step_t), Python: int(np.ceil((t+1)/step_t)) - 1
        idx_time = int(np.ceil((t + 1) / step_t)) - 1

        dia_lay = diapause_lay.values[:, :, idx_time]
        dia_hatch = diapause_hatch.values[:, :, idx_time]
        ed_surv = ed_survival[:, :, t]
        water_hatch = egg_activate.values[:, :, idx_time]
        mort_e = mosq_mort_e(T)
        mort_j = mosq_mort_j(T)
        Tmean_slice = Tmean.values[:, :, idx_time]
        mort_a = mosq_mort_a(Tmean_slice)

        vars_tuple = (
            idx_time + 1,  # Octave uses 1-based, so pass idx_time+1
            step_t,
            Temp,
            CC,
            birth,
            dia_lay,
            dia_hatch,
            mort_e,
            mort_j,
            mort_a,
            ed_surv,
            dev_j,
            dev_i,
            dev_e,
            water_hatch,
        )

        va = rk4_step(eqsys, eqsys_log, v, vars_tuple, step_t)
        break

        # # Zero compartment 2 (Python index 1) if needed
        # if (t / step_t) % 365 == 200:
        #     v[..., 1] = 0

        # # Store output every step_t
        # if t % step_t == 0:
        #     if ((idx_time + 1) % 30) == 0:
        #         print(f"MOY: {(idx_time + 1) / 30}")
        #     for j in range(5):
        #         v_out[..., j, idx_time] = np.maximum(v[..., j], 0)

    return va


def mosq_mort_e(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_mort_e function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with mortality rates applied elementwise
    """
    # TODO: Status --> it works!

    T_out = 0.955 * np.exp(-0.5 * ((T - 18.8) / 21.53) ** 6)
    T_out = -np.log(T_out)
    return T_out


def mosq_mort_j(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_mort_j function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with juvenile mortality rates applied elementwise
    """
    # TODO: Status --> it works

    T_out = 0.977 * np.exp(-0.5 * ((T - 21.8) / 16.6) ** 6)
    T_out = np.where(T_out > 0, T_out, 1e-12)
    T_out = -np.log(T_out)
    return T_out


def mosq_mort_a(T: np.ndarray) -> np.ndarray:
    """
    Python implementation of the Octave mosq_mort_a function.

    Args:
        T: numpy.ndarray of temperatures

    Returns:
        numpy.ndarray with adult mortality rates applied elementwise
    """
    # TODO: Status --> it works
    T_out = T.copy()
    mask_pos = T_out > 0
    mask_zero = ~mask_pos

    # For T > 0
    T_out[mask_pos] = (
        0.677
        * np.exp(-0.5 * ((T_out[mask_pos] - 20.9) / 13.2) ** 6)
        * T_out[mask_pos] ** 0.1
    )
    # For T <= 0
    T_out[mask_zero] = 0.677 * np.exp(-0.5 * ((T_out[mask_zero] - 20.9) / 13.2) ** 6)
    T_out = np.where(T_out > 0, T_out, 1e-12)
    T_out = -np.log(T_out)
    return T_out


def eqsys(v, vars):
    # Unpack variables
    (
        t_idx,
        step_t,
        Temp,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = vars

    FT = np.zeros_like(v)

    # Differential equations (vectorized over grid)
    # Egg compartment (non-diapause)
    FT[..., 0] = (
        v[..., 4] * birth * (1 - dia_lay)  # Oviposition (non-diapause)
        - (mort_e + water_hatch * dev_e) * v[..., 0]  # Mortality and hatching
    )

    # Egg compartment (diapause)
    FT[..., 1] = (
        v[..., 4] * birth * dia_lay  # Oviposition (diapause)
        - water_hatch * dia_hatch * v[..., 1]  # Hatching from diapause
    )

    # Juvenile compartment
    FT[..., 2] = (
        water_hatch * dev_e * v[..., 0]  # Hatching from non-diapause eggs
        + water_hatch * dia_hatch * ed_surv * v[..., 1]  # Hatching from diapause eggs
        - (mort_j + dev_j) * v[..., 2]  # Mortality and development
        - (v[..., 2] ** 2) / CC[..., t_idx - 1]  # Density-dependent mortality
    )

    # Immature adult compartment
    FT[..., 3] = (
        0.5 * dev_j * v[..., 2]  # Development from juveniles
        - (mort_a + dev_i) * v[..., 3]  # Mortality and maturation
    )

    # Mature adult compartment
    FT[..., 4] = (
        dev_i * v[..., 3]  # Maturation from immature adults
        - mort_a * v[..., 4]  # Adult mortality
    )

    # Replace NaNs
    FT[np.isnan(-FT)] = -v[np.isnan(-FT)] * step_t
    return FT


def eqsys_log(v, vars):
    # Unpack variables
    (
        t_idx,
        step_t,
        Temp,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    ) = vars

    FT = np.zeros_like(v)
    FT[..., 0] = v[..., 4] * birth * (1 - dia_lay) / v[..., 0] - (
        mort_e + water_hatch * dev_e
    )
    FT[..., 1] = v[..., 4] * birth * dia_lay / v[..., 1] - water_hatch * dia_hatch
    FT[..., 2] = (
        water_hatch * dev_e * v[..., 0] / v[..., 2]
        + water_hatch * dia_hatch * ed_surv * v[..., 1] / v[..., 2]
        - (mort_j + dev_j)
        - v[..., 2] / CC[..., t_idx - 1]
    )
    FT[..., 3] = 0.5 * dev_j * v[..., 2] / v[..., 3] - (mort_a + dev_i)
    FT[..., 4] = dev_i * v[..., 3] / v[..., 4] - mort_a
    FT[np.isnan(-FT)] = -v[np.isnan(-FT)] * step_t
    return FT


def rk4_step(f, flog, v, vars, step_t):
    # Octave-style RK4 with negative value correction using log-form ODEs
    k1 = f(v, vars)
    k2 = f(v + 0.5 * k1 / step_t, vars)
    k3 = f(v + 0.5 * k2 / step_t, vars)
    k4 = f(v + k3 / step_t, vars)
    v1 = v + (k1 + 2 * k2 + 2 * k3 + k4) / (step_t * 6)

    # Check for negative values in all RK4 steps
    neg_mask = (
        (v1 < 0)
        | ((v + 0.5 * k1 / step_t) < 0)
        | ((v + 0.5 * k2 / step_t) < 0)
        | ((v + k3 / step_t) < 0)
    )

    if np.any(neg_mask):
        v2 = np.log(np.clip(v, 1e-12, None))  # avoid log(0)
        FT2 = flog(v2, vars)
        v2 = v2 + FT2 / step_t
        v1[neg_mask] = np.exp(v2[neg_mask])

    return v1


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

    def print_slices_numpy(arr):
        """
        Print each slice along the last axis of a 3D numpy array.
        """
        for i in range(arr.shape[2]):
            print(f"Slice at index {i}:")
            print(arr[:, :, i])
            print()  # Blank line for readability

    # diapause_lay = mosq_dia_lay(
    #     T=model_data.temperature_mean,
    #     LATU=model_data.latitude,
    # )
    # print_slices(diapause_lay, 3)

    # diapause_hatch = mosq_dia_hatch(
    #     T=model_data.temperature_mean,
    #     LATU=model_data.latitude,
    #     step_t=10,
    # )

    # print_slices(diapause_hatch, 3)

    # ed_survival = mosq_surv_ed(T=model_data.temperature)
    # print(print_slices_numpy(ed_survival))

    # print("Temperatures")
    # print(model_data.temperature)

    # TODO: rethink strategy for storing inputs and outputs objects
    # step_t = 10
    # shape_output = (
    #     model_data.initial_conditions.shape[0],
    #     model_data.initial_conditions.shape[1],
    #     5,
    #     int(model_data.temperature.shape[2] / step_t),
    # )
    # print(shape_output)

    # print(model_data.temperature[:, :, 0])
    # # print(mosq_birth(model_data.temperature[:, :, 0]))

    # # print(mosq_dev_j(model_data.temperature[:, :, 0]))

    # print(mosq_dev_i(model_data.temperature[:, :, 0]))

    # # TODO: ask about how to organize this into the fucntion `mosq_dev_e`
    # dev_e = 1.0 / 7.1
    # print(mosq_dev_e(model_data.temperature[:, :, 0]))

    CC = carrying_capacity(
        rainfall_data=model_data.rainfall, population_data=model_data.population_density
    )

    egg_activate = water_hatching(
        rainfall_data=model_data.rainfall, population_data=model_data.population_density
    )

    va = for_cycle_equation(
        v=model_data.initial_conditions,
        Temp=model_data.temperature,
        Tmean=model_data.temperature_mean,
        LAT=model_data.latitude,
        CC=CC,
        egg_activate=egg_activate,
        step_t=10,
    )

    print_slices_numpy(va)
    print(va[:, :, 0])

    print_slices_numpy(model_data.initial_conditions)
