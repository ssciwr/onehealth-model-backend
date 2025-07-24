import numpy as np
import xarray as xr
from scipy.interpolate import interpn

# -----------------------------
# 1. Your datasets
# -----------------------------
# pr dataset (reference grid)
lon1 = np.array([-180.0, -179.5, -179.0, -178.5, -178.0])
lat1 = np.array([-89.5, -89.0, -88.5, -88.0])
time1 = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")

pr_data = np.array(
    [
        [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20],
        ],
        [
            [21, 22, 23, 24, 25],
            [26, 27, 28, 29, 30],
            [31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40],
        ],
    ]
)
pr = xr.Dataset(
    {"tp": (["time", "latitude", "longitude"], pr_data)},
    coords={"longitude": lon1, "latitude": lat1, "time": time1},
)

# dens dataset (different grid)
lon2 = np.array([-179.75, -179.25, -178.75, -178.25, -177.75])
lat2 = np.array([-89.75, -89.25, -88.75, -88.25])
time2 = np.array(["2024-01-01"], dtype="datetime64[ns]")

dens_data = np.array(
    [
        [
            [-1, -2, -3, -4, -5],
            [-6, -7, -8, -9, -10],
            [-11, -12, -13, -14, -15],
            [-16, -17, -18, -19, -20],
        ]
    ]
)
dens = xr.Dataset(
    {"dens": (["time", "latitude", "longitude"], dens_data)},
    coords={"longitude": lon2, "latitude": lat2, "time": time2},
)

# -----------------------------
# 2. Interpolation with xarray
# -----------------------------
dens_interp_xr = dens.interp(
    longitude=pr.longitude, latitude=pr.latitude, method="linear"
)

# -----------------------------
# 3. Interpolation with scipy
# -----------------------------
# For simplicity, interpolate dens[0] to pr grid
points = (lat2, lon2)  # Grid of dens
values = dens.dens[0, :, :].values  # Shape: (lat, lon)

# Meshgrid of pr target coordinates
lat_mesh, lon_mesh = np.meshgrid(pr.latitude.values, pr.longitude.values, indexing="ij")
query_points = np.column_stack((lat_mesh.ravel(), lon_mesh.ravel()))

# Interpolate
dens_interp_scipy_flat = interpn(
    points, values, query_points, method="linear", bounds_error=False, fill_value=np.nan
)

# Reshape to (lat, lon)
dens_interp_scipy = dens_interp_scipy_flat.reshape(len(pr.latitude), len(pr.longitude))

# -----------------------------
# 4. Compare results
# -----------------------------
print("Interpolated values (xarray):")
print(dens_interp_xr.dens[0].values)

print("\nInterpolated values (scipy):")
print(dens_interp_scipy)

# Difference
diff = dens_interp_xr.dens[0].values - dens_interp_scipy
print("\nDifference (xarray - scipy):")
print(diff)

print("\nMax absolute difference:", np.nanmax(np.abs(diff)))
