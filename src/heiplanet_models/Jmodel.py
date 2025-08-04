from __future__ import annotations

import xarray as xr
import pandas as pd
import dask.array as da
from pathlib import Path
from .utils import read_geodata, detect_csr
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class JModelData:
    name: str
    input: str | Path | None = None  # Placeholder for input data, to be
    output: str | Path | None = None  # Placeholder for output data, to be defined later
    run_mode: str = "forbidden"  # dask run mode used by xarray
    r0_data: pd.DataFrame | None = None  # Placeholder for R0 data
    min_temp: float = 0.0  # Minimum temperature for interpolation
    max_temp: float = 45.0  # Maximum temperature for interpolation
    step: float = 0.1  # Step size for temperature interpolation
    temp_colname: str = "t2m"
    out_colname: str = "R0"
    grid_data_baseurl: str | None = None
    nuts_level: int | None = None  # NUTS level for the model, default is 0
    resolution: str | None = None  # Resolution for the nuts data
    year: int | None = None  # Year for the model


def read_default_config() -> dict[str, str | np.int64 | None]:
    """Reads the default configuration for the JModel from a JSON file.

    Returns:
        dict[str, str | np.int64 | None]: A dictionary containing the default configuration.
    """
    config_path = Path(__file__).parent / "config_Jmodel.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def setup_modeldata(
    input: str | None = None,
    output: str | None = None,
    r0_path: str | None = None,
    run_mode: str = "forbidden",
    grid_data_baseurl: str | None = None,
    nuts_level: int | None = None,
    resolution: str | None = None,
    year: int | None = None,
    temp_colname: str = "t2m",
    out_colname: str = "R0",
) -> JModelData:
    """Initializes the JModel with the given configuration.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the model.
    """

    # set up plumbing for the model
    if run_mode not in ["forbidden", "parallelized"]:
        raise ValueError(
            f"Invalid run mode: {run_mode}. Supported modes are 'forbidden', 'parallelized'. For the meaning of these modes, see the documentation. of xarray.apply_ufunc"
        )

    # set data paths and get r0 data
    if input is None:
        raise ValueError("Input data path must be provided in the configuration.")

    if output is None:
        raise ValueError("Output data path must be provided in the configuration.")

    # read R0 data from the given path
    if r0_path is None:
        raise ValueError("R0 data path must be provided in the configuration.")
    else:
        r0_data = pd.read_csv(r0_path)
    step_temp = (
        r0_data.Temperature[1] - r0_data.Temperature[0]
    )  # assume uniform step size
    min_temp = r0_data.Temperature.min()
    max_temp = r0_data.Temperature.max()

    if any(
        [
            grid_data_baseurl is None,
            nuts_level is None,
            resolution is None,
            year is None,
        ]
    ) and not all(
        [
            grid_data_baseurl is None,
            nuts_level is None,
            resolution is None,
            year is None,
        ]
    ):
        raise ValueError(
            "Grid data configuration is incomplete. Please provide all parameters: grid_data_baseurl, nuts_level, resolution, and year, or do not set any to have them all set to 'None'."
        )
    else:
        # don´t do anything here, because None indicates the grid data is not used
        pass

    return JModelData(
        name="JModel",
        input=input,
        output=output,
        run_mode=run_mode,
        r0_data=r0_data,
        min_temp=min_temp,
        max_temp=max_temp,
        step=step_temp,
        temp_colname=temp_colname,
        out_colname=out_colname,
        grid_data_baseurl=grid_data_baseurl,
        nuts_level=nuts_level,
        resolution=resolution,
        year=year,
    )


def _interpolate_r0(
    temp: np.ndarray | da.core.Array,
    r0_data: pd.DataFrame,
    min_temp: float,
    max_temp: float,
) -> np.ndarray | da.core.Array:
    """Interpolates R0 values based on temperature using the stored R0 data.
    Args:
        temp (np.ndarray | da.core.Array): Temperature values for which to interpolate R0.
        r0_data (pd.DataFrame): DataFrame containing R0 data with 'Temperature'
            and 'Median_R0' columns.
        min_temp (np.float64): Minimum temperature for interpolation.
        max_temp (np.float64): Maximum temperature for interpolation.
    Returns:
        np.ndarray | da.core.Array: Interpolated R0 values corresponding to the input temperature values.
    """
    temp_values = temp.compute() if hasattr(temp, "compute") else temp
    if not isinstance(temp_values, np.ndarray) and not isinstance(temp, da.core.Array):
        temp_values = temp_values.values  # Ensure temp is a numpy array

    # Create result array with same shape as input
    result = np.full_like(temp, np.nan, dtype=float)

    # Find valid temperature values (equivalent to R's valid mask)
    valid_mask = (
        ~np.isnan(temp_values) & (temp_values >= min_temp) & (temp_values <= max_temp)
    )

    # Only interpolate where we have valid values
    if np.any(valid_mask):
        result[valid_mask] = np.interp(
            temp_values[valid_mask],  # Only pass valid values
            r0_data.Temperature.values,
            r0_data.Median_R0.values,
            left=np.nan,
            right=np.nan,
        )

    return result


def read_input_data(model_data: JModelData) -> xr.Dataset:
    """Read input data from given source 'model_data.input'

    Args:
        model_data (JModelData): Data class containing the model configuration and input data path.

    Returns:
        xr.Dataset: xarray dataset containing the input data for the model.
    """

    # nothing done here yet
    data = xr.open_dataset(
        model_data.input, chunks=None if model_data.run_mode == "forbidden" else "auto"
    )

    if data is None:
        raise ValueError("Input data source is not defined in the configuration.")

    # ensure the data has a coordinate reference system (CRS)
    data = detect_csr(data)

    # read the grid data if we want to crop the data
    if all(
        [
            model_data.grid_data_baseurl is not None,
            model_data.nuts_level is not None,
            model_data.resolution is not None,
            model_data.year is not None,
        ]
    ):
        grid_data = read_geodata(
            base_url=model_data.grid_data_baseurl,
            nuts_level=model_data.nuts_level,
            resolution=model_data.resolution,
            year=model_data.year,
            url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
        )

        if grid_data.crs != data.rio.crs:
            raise ValueError(
                f"Coordinate reference system mismatch: Grid data CRS {grid_data.crs} does not match input data CRS {data.rio.crs}."
            )

        # crop the data to the grid. This will remove the pixels outside the grid area
        data = data.rio.clip(
            grid_data.geometry.values,
            grid_data.crs,
            drop=True,  # Drop pixels outside the clipping area
        )

    if model_data.run_mode == "forbidden":
        # run synchronously on one cpu
        return data.compute()
    else:
        return data


def run_model(
    model_data: JModelData, data: xr.Dataset | pd.DataFrame
) -> xr.Dataset | pd.DataFrame:
    """Runs the JModel with the provided input data. Applies the R0 interpolation based on temperature values from the stored R0 data and returns a new dataset or dataframe with the R0 data.

    Args:
        model_data (JModelData): _description_
        data (xr.Dataset | pd.DataFrame): _description_

    Returns:
        xr.Dataset | pd.DataFrame: A dataset or dataframe with the incoming R0 data interpolated based on the temperature values at each grid point.
    """
    r0_map = xr.apply_ufunc(
        lambda t: _interpolate_r0(
            t,
            model_data.r0_data,
            model_data.min_temp,
            model_data.max_temp,
        ),
        data[model_data.temp_colname],
        input_core_dims=[[]],
        output_core_dims=[[]],
        dask=model_data.run_mode,
        keep_attrs=True,
    ).rename(model_data.out_colname)

    return r0_map


def store_output_data(
    model_data: JModelData, data: xr.Dataset | xr.DataArray | pd.DataFrame
) -> None:
    data.to_netcdf(model_data.output)
    data.close()  # Close the dataset to free resources
