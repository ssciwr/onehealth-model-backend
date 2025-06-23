from __future__ import annotations

import xarray as xr
import pandas as pd
import dask.array as da
from pathlib import Path
from .utils import read_geodata, detect_csr
import numpy as np


class JModel:
    """
    Implements the model used in the paper *TODO*
    """

    name: str = "JModel"
    input: str | Path | None = None  # Placeholder for input data, to be defined later
    output: str | Path | None = None  # Placeholder for output data, to be defined later
    run_mode: str = "forbidden"
    r0_data: pd.DataFrame  # Placeholder for R0 data, to be defined later
    min_temp: np.float64 = 0.0  # Minimum temperature for interpolation
    max_temp: np.float64 = 45.0  # Maximum temperature for interpolation
    step: np.float64 = 0.1  # Step size for temperature interpolation
    temp_colname: str = "t2m"
    out_colname: str = "R0"
    grid_data_baseurl: str | None = None
    nuts_level: int | None = None  # NUTS level for the model, default is 0
    resolution: str | None = None  # Resolution for the nuts data
    year: int | None = None

    def __init__(
        self,
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
    ):
        """Initializes the JModel with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """

        # set up plumbing for the model
        self.run_mode = run_mode

        if self.run_mode not in ["forbidden", "parallelized"]:
            raise ValueError(
                f"Invalid run mode: {self.run_mode}. Supported modes are 'forbidden', 'parallelized'. For the meaning of these modes, see the documentation. of xarray.apply_ufunc"
            )

        # set data paths and get r0 data
        if input is not None:
            self.input = input
        else:
            raise ValueError("Input data path must be provided in the configuration.")

        if output is not None:
            self.output = output
        else:
            raise ValueError("Output data path must be provided in the configuration.")

        # read R0 data from the given path
        if r0_path is None:
            raise ValueError("R0 data path must be provided in the configuration.")
        else:
            self.r0_data = pd.read_csv(r0_path)
        self.step_temp = (
            self.r0_data.Temperature[1] - self.r0_data.Temperature[0]
        )  # assume uniform step size
        self.min_temp = self.r0_data.Temperature.min()
        self.max_temp = self.r0_data.Temperature.max()

        self.temp_colname = temp_colname
        self.out_colname = out_colname

        if all(
            [
                grid_data_baseurl is not None,
                nuts_level is not None,
                resolution is not None,
                year is not None,
            ]
        ):
            self.grid_data_baseurl = grid_data_baseurl
            self.nuts_level = nuts_level
            self.resolution = resolution
            self.year = year
        elif any(
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

    def _interpolate_r0(
        self, temp: np.ndarray | da.core.Array
    ) -> np.ndarray | da.core.Array:
        """Interpolates R0 values based on temperature using the stored R0 data.
        Args:
            temp (np.ndarray | da.core.Array): Temperature values to interpolate R0 for.
        Returns:
            np.ndarray | da.core.Array: Interpolated R0 values corresponding to the input temperature values.
        """
        temp_values = temp.compute() if hasattr(temp, "compute") else temp
        if not isinstance(temp_values, np.ndarray) and not isinstance(
            temp, da.core.Array
        ):
            temp_values = temp_values.values  # Ensure temp is a numpy array

        # Create result array with same shape as input
        result = np.full_like(temp, np.nan, dtype=float)

        # Find valid temperature values (equivalent to R's valid mask)
        valid_mask = (
            ~np.isnan(temp_values)
            & (temp_values >= self.min_temp)
            & (temp_values <= self.max_temp)
        )

        # Only interpolate where we have valid values
        if np.any(valid_mask):
            result[valid_mask] = np.interp(
                temp_values[valid_mask],  # Only pass valid values
                self.r0_data.Temperature.values,
                self.r0_data.Median_R0.values,
                left=np.nan,
                right=np.nan,
            )

        return result

    def read_input_data(self) -> xr.Dataset:
        """Read input data from given source 'self.input'

        Returns:
            xr.Dataset: xarray dataset containing the input data for the model.
        """

        # nothing done here yet
        data = xr.open_dataset(
            self.input, chunks=None if self.run_mode == "forbidden" else "auto"
        )

        if data is None:
            raise ValueError("Input data source is not defined in the configuration.")

        # ensure the data has a coordinate reference system (CRS)
        data = detect_csr(data)

        # read the grid data if we want to crop the data
        if all(
            [
                self.grid_data_baseurl is not None,
                self.nuts_level is not None,
                self.resolution is not None,
                self.year is not None,
            ]
        ):
            grid_data = read_geodata(
                base_url=self.grid_data_baseurl,
                nuts_level=self.nuts_level,
                resolution=self.resolution,
                year=self.year,
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

        if self.run_mode == "forbidden":
            # run synchronously on one cpu
            return data.compute()
        else:
            return data

    def run(
        self,
    ) -> None:
        """Runs the JModel with the provided input data. Reads the data first, then applies the R0 interpolation based on temperature values from the stored R0 data, finally writes the result back to file."""

        with self.read_input_data() as data:
            r0_map = xr.apply_ufunc(
                self._interpolate_r0,
                data[self.temp_colname],
                input_core_dims=[[]],
                output_core_dims=[[]],
                dask=self.run_mode,
                keep_attrs=True,
            ).rename(self.out_colname)

            self.store_output_data(r0_map)

    def store_output_data(self, data: xr.Dataset) -> None:
        """Stores the processed data to the specified output file.
        *for now, this is netcdf4, later we will have a database connection for this*

        Args:
            data (xr.Dataset): Dataset to store
        """
        data.to_netcdf(self.output)
