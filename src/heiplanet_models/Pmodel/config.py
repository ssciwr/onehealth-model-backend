import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# ============================================
# ====              DATASETS              ====
# ============================================
# ---- Root folder where all the NetCDF files will be stored
PATH_ROOT = Path("data/in/Pratik_datalake")

DATASET_PREFIXES = {
    "TEMPERATURE":"ERA5land_global_t2m_daily_0.5_",
    "RAINFALL" : "ERA5land_global_tp_daily_0.5_",
    "POPULATION_DENSITY" : "pop_dens_",
}

SETTINGS_DATASET= {
    "TEMPERATURE":{
        "decode_times": False,
        "names_dimensions":{"valid_time: time"},    
        "dimension_order": ("longitude", "latitude", "time"),
        "variable_name": "t2m",
    },
    "RAINFALL":{
        "names_dimensions": None,
        "dimension_order": ("longitude", "latitude", "time"),
        "variable_name": "tp",
    },
    "POPULATION":{
        "decode_times": True,
        "names_dimensions":{"lon": "longitude", "lat": "latitude", "time": "time"},
        "dimension_order": ("longitude", "latitude", "time"),
        "variable_name":"dens"

    }

}

# # ---- Paths to files
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
