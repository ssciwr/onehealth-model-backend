from dataclasses import dataclass

import xarray as xr


@dataclass
class PmodelInput:
    var_temperature: xr.Dataset
    var_rainfall: xr.Dataset
    var_human_population: xr.Dataset
