import numpy as np 
import xarray as xr

# basic type for data that can be processed by the model
type oneData = xr.Dataset | xr.DataArray | np.ndarray