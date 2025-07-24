from pathlib import Path

import numpy as np
import xarray as xr

import Pmodel_params


def capacity(pr: xr.DataArray, dens: xr.DataArray, **kwargs) -> xr.DataArray:

    # Constants
    ALPHA = kwargs.get("ALPHA", Pmodel_params.ALPHA_RAIN)
    BETA = kwargs.get("BETA", Pmodel_params.ALPHA_DENS)
    GAMMA = kwargs.get("GAMMA", Pmodel_params.GAMMA)
    LAMBDA = kwargs.get("LAMBDA", Pmodel_params.LAMBDA)

    # TODO: Remove the following section for production

    pr_new = pr.copy(deep=True)  # Keep
    print(f"pr dims: {pr.dims}")
    print(f"dens dims: {dens.dims}")
    print(pr.coords)
    print(dens.coords)

    # Select first time slice
    pr_slice = pr.isel(time=0)
    print("pr values: \n{}".format(pr_slice.values))

    # Select first time slice in dens
    dens_slice = dens.isel(time=0)
    print("dens values: \n{}".format(dens_slice.values))

    # Now sum
    # result = pr_slice + dens_slice

    # print(result.shape)
    # print("Result of sum pr + dense: {}".format(result.compute().values))

    # Get time length
    time = pr.coords["time"]
    ntime = len(time)

    # Initialize A(t)
    pr_new.loc[dict(time=time[0])] = ALPHA * pr_new.sel(time=time[0]) + BETA * dens.sel(
        time=time[0]
    )

    # Apply recursive formula for A(t)
    for k in range(1, ntime):
        prev = pr_new.sel(time=time[k - 1])
        curr_pr = pr.sel(time=time[k])
        curr_dens = dens.sel(time=time[0])

        pr_new.loc[dict(time=time[k])] = GAMMA * prev + ALPHA * curr_pr + BETA * curr_dens

    # Apply scaling factor to compute K(t)
    for k in range(1, ntime):
        factor = (1 - GAMMA) / (1 - GAMMA ** (k + 1))
        pr_new.loc[dict(time=time[k])] = factor * pr_new.sel(time=time[k])

    # Final scaling
    return pr_new * LAMBDA


if __name__ == "__main__":
    from Pmodel_initial import load_data

    model_data = load_data(time_step=10)
    model_data.print_attributes()

    constants_capacity = {
        "ALPHA": 10,  # 1e-3
        "BETA": 10,  # 1e-5
        "GAMMA": 10,  # 9e-1
        "LAMBDA": 10,  # 1e6 * 625 * 100
    }

    CC = capacity(
        pr=model_data.rainfall,
        dens=model_data.population_density,
        **constants_capacity,
    )

    for i in range(3):  # for indices 0, 1, 2
        print(f"Slice at time index {i}:")
        print(CC.isel(time=i).values)
        print()  # Blank line for readability
