from dataclasses import dataclass
from math import ceil

import numpy as np
import xarray as xr


@dataclass
class PmodelInitial:
    def __init__(self):
        pass


def load_human_population_density(human_population):
    """

    Ref. `load_hdp.m`
    """
    return human_population


def load_initial():
    pass


def load_latitude(filepath_dataset, variable_name="latitude"):
    latitude = xr.open_dataset(filepath_dataset)[variable_name]
    return latitude


def load_rainfall(rainfall):
    """

    Ref. `load_rainfall.m`
    """

    return rainfall


def load_temperature(temperature_mean, step_size):
    """

    Ref. `load_temp2.m`

    """

    # (time, lat., long) <- 3D data
    valid_time, latitude, longitude = temperature_mean.shape

    T = np.zeros((valid_time * step_size, latitude, longitude))

    for time in range(0, valid_time * step_size):
        td = int(np.ceil((time + 1) / step_size)) - 1  # Adjust for Python 0-based index
        T[time, :, :] = temperature_mean[td, :, :]

    ## Vectorized implementation
    ## Precompute coarse indices for each fine-grained timestep
    # td_indices = np.ceil((np.arange(1, valid_time * step_size + 1) / step_size)).astype(int) - 1

    ## Index temperature_mean using td_indices along time axis
    # T = temperature_mean[td_indices, :, :]

    return T, temperature_mean


def load_data() -> PmodelInitial:
    pass


def juvenile_carrying_capacity(rainfall, population_density):
    """

    Ref. `capacity.m`
    Ref. supplementary material, eq. 14

    $$
    K_{L}(W,P)) (W)
    $$

    """
    ALPHA = 0.001
    BETA = 0.00001
    GAMMA = 0.9

    LAMBDA = 1e6 * 625 * 100

    pr = rainfall
    dens = population_density

    # Create a copy to avoid in-place modification
    pr_new = pr.copy()

    print(f"PR shape (time, rows, cols): {pr.shape}")
    print(f"DENS shape (rows, cols): {dens.shape}")

    print(pr_new.values)

    # Update for time index 0
    # pr_new[0, :, :] = ALPHA * pr_new[0, :, :] + BETA * dens

    # # Recursive update for time >= 1
    # for k in range(1, pr_new.shape[0]):  # time is first dimension now
    #     pr_new[k, :, :] = (
    #         GAMMA * pr_new[k - 1, :, :]  # previous time slice
    #         + ALPHA * pr_new[k, :, :]  # current time slice
    #         + BETA * dens  # dens contribution
    #     )

    # # Normalization
    # for k in range(1, pr_new.shape[0]):
    #     scaling_factor = (1 - GAMMA) / (1 - GAMMA ** (k + 1))  # +1 since k is 0-based
    #     pr_new[k, :, :] *= scaling_factor

    # # Apply scaling
    # pr_new *= LAMBDA

    # return pr_new
