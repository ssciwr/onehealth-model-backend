from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class PmodelInput:
    initial_conditions: np.ndarray = None
    latitude: xr.DataArray
    population_density: xr.DataArray
    rainfall: xr.DataArray
    temperature: xr.DataArray
    temperature_mean: xr.DataArray
