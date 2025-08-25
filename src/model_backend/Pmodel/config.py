import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PATH_DATASETS_SANDBOX = {
    "TEMPERATURE": Path("data/in/Pratik_datalake/temperature_dummy.nc"),
    "RAINFALL": Path("data/in/Pratik_datalake/pr_dummy.nc"),
    "HUMAN_POPULATION": Path("data/in/Pratik_datalake/dense_dummy.nc"),
}

PATH_DATASETS_PRODUCTION = {
    "TEMPERATURE": Path("data/in/Pratik_datalake/ERA5land_global_t2m_daily_0.5_2024.nc"),
    "RAINFALL": Path("data/in/Pratik_datalake/ERA5land_global_tp_daily_0.5_2024.nc"),
    "HUMAN_POPULATION": Path("data/in/Pratik_datalake/pop_dens_2024_global_0.5.nc"),
}
