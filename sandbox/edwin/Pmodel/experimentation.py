import xarray as xr

era5_t2m_path = "/home/ecarreno/SSC-Projects/b_REPOSITORIES/onehealth-model-backend/data/in/ERA5land_global_tp_daily_0.5_2024.nc"
population_path = "/home/ecarreno/SSC-Projects/b_REPOSITORIES/onehealth-model-backend/data/in/pop_dens_2024_global_0.5.nc"


pr = xr.open_dataset(era5_t2m_path)["tp"]
dense = xr.open_dataset(population_path)["total-population"]


# Dimensions
print(f"Dimensions pr: {pr.dims}")
print(f"Dimensions dense: {dense.dims}")


# Variables
print(dense.shape)
print(pr[0, :, :].shape)

dense[0, :, :] + pr[0, :, :]
