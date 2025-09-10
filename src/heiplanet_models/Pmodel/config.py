import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================
# ====              DATASETS              ====
# ============================================
# ---- Root folder where all the NetCDF files will be stored
PATH_ROOT = Path("data/in/Pratik_datalake")

# ---- Paths to files
PATH_DATASETS_SANDBOX = {
    "TEMPERATURE": Path("data/in/Pratik_datalake/temperature_dummy.nc"),
    "RAINFALL": Path("data/in/Pratik_datalake/pr_dummy.nc"),
    "HUMAN_POPULATION": Path("data/in/Pratik_datalake/dense_dummy.nc"),
}

PATH_DATASETS_PRODUCTION = {
    "TEMPERATURE": Path(
        "data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"
    ),
    "RAINFALL": Path("data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"),
    "HUMAN_POPULATION": Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc"),
}

DATASET_PREFIXES = {
    "TEMPERATURE_DATASET":"ERA5land_global_t2m_daily_0.5_",
    "RAINFALL_DATASET" : "ERA5land_global_tp_daily_0.5_",
    "POPULATION_DENSITY_DATASET" : "pop_dens_",
}

# ---- Configuration initial conditions
CONST_K1 = 625
CONST_K2 = 100

# ---- Configuration load data
CHUNKING_SCHEME = {
    "longitude": 90,
    "latitude": 45,
    "time": 10,
}
COORDINATES_ORDER = ("longitude", "latitude", "time")

# ---- ODE
MODEL_VARIABLES = ["eggs", "ed", "juv", "imm", "adults"]
TIME_STEP = 10


DATASET_PREFIXES = {
    "TEMPERATURE_DATASET":"ERA5land_global_t2m_daily_0.5_",
    "RAINFALL_DATASET" : "ERA5land_global_tp_daily_0.5_",
    "POPULATION_DENSITY_DATASET" : "pop_dens_",
}