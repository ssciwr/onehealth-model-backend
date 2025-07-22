from pathlib import Path

import numpy as np
import xarray as xr

from Pmodel_params import (
    ALPHA_DENS,
    ALPHA_RAIN,
    GAMMA,
    LAMBDA,
)


def capacity(pr: xr.DataArray, dens: xr.DataArray) -> xr.DataArray:

    # Constants
    ALPHA = 0.001
    BETA = 0.00001
    GAMMA = 0.9
    LAMBDA = 1e6 * 625 * 100

    pr_new = pr.copy(deep=True)
    print(f"pr dims: {pr.dims}")
    print(f"dens dims: {dens.dims}")
    print(pr.coords)
    print(dens.coords)

    # Select first time slice
    pr_slice = pr.isel(time=0)
    print(pr_slice.values)

    # dens_slice = dens.isel(time=0)

    # SUPER REALLY ULTRA IMPORTANT
    # Reindex dens to match pr's longitude and latitude
    # dens_aligned = dens_slice.interp(
    #     longitude=pr_slice.longitude,
    #     latitude=pr_slice.latitude,
    #     method="nearest",  # or "linear" for interpolation
    # )

    # # Now sum
    # result = pr_slice + dens_aligned

    # print(result.compute().shape)
    # print(result)

    # print(result.values[0, 0])
    # Data usage
    # print(pr.data.shape)
    # print(pr_new.data.shape)
    # print(dens.data.shape)
    # print(a.compute().shape)
    # print(pr_new.isel(time=0).shape)
    # print(dens.isel(time=0).shape)

    return None


if __name__ == "__main__":
    from Pmodel_initial import load_data

    model_data = load_data(time_step=10)
    model_data.print_attributes()

    capacity(pr=model_data.rainfall, dens=model_data.population_density)

    # Approach 2. Using pure functions
    # PATH_DATASET_TEMPERATURE = Path(
    #     "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
    # )
    # PATH_DATASET_RAINFALL = Path(
    #     "data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"
    # )
    # PATH_DATASET_POPULATION = Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc")

    # pr = load_rainfall(path_rainfall_dataset=PATH_DATASET_RAINFALL, variable_name="tp")
    # dens = load_human_population_density(
    #     path_population_dataset=PATH_DATASET_POPULATION, variable_name="dens"
    # )

    # capacity = capacity(pr=pr, dens=dens)

    # print(capacity.shape)
