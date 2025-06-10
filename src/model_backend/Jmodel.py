import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
from .utils import read_geodata, detect_csr

type oneData = xr.Dataset | xr.DataArray | np.ndarray


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

    grid_data_baseurl: str | None = None
    nuts_level: int = 0  # NUTS level for the model, default is 0
    resolution: str = "10M"  # Resolution for the nuts data
    year: int = 2024

    def __init__(
        self,
        input: str | None = None,
        output: str | None = None,
        r0_path: str | None = None,
        run_mode: str = "forbidden",
        grid_data_baseurl: str | None = None,
        nuts_level: int = 3,
        resolution: str = "10M",
        year: int = 2024,
    ):
        """Initializes the JModel with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """

        # set up plumbing for the model
        self.run_mode = run_mode

        if self.run_mode not in ["allowed", "forbidden", "parallelized"]:
            raise ValueError(
                f"Invalid run mode: {self.run_mode}. Supported modes are 'allowed', 'forbidden', 'parallelized'."
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
            self.r0_data = pd.read_csv(r0_path) if r0_path else None
        self.step_temp = (
            self.r0_data.Temperature[1] - self.r0_data.Temperature[0]
        )  # assume uniform step size
        self.min_temp = self.r0_data.Temperature.min()
        self.max_temp = self.r0_data.Temperature.max() + self.step_temp

        if grid_data_baseurl is not None:
            self.grid_data_baseurl = grid_data_baseurl
        else:
            raise ValueError(
                "Grid data base URL must be provided in the configuration."
            )

        self.nuts_level = nuts_level
        self.resolution = resolution
        self.year = year

    def read_input_data(self) -> oneData:
        """Read input data from given source 'self.input'

        Returns:
            oneData: xarray dataset or data array containing the input data for the model.
        """

        # nothing done here yet
        data = xr.open_dataset(self.input, engine="rasterio", chunks="auto")

        if data is None:
            raise ValueError("Input data source is not defined in the configuration.")

        # read grid data, set coordinate reference system (CRS) if not set and then crop the data
        # will be thrown away again after this function is done
        grid_data = read_geodata(
            base_url=self.grid_data_baseurl,
            nuts_level=self.nuts_level,
            resolution=self.resolution,
            year=self.year,
            url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
        )

        # ensure the data has a coordinate reference system (CRS)
        data = detect_csr(data)

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

        return data

    def run(
        self,
    ) -> None:
        """Runs the JModel with the provided input data. Reads the data first, then applies the R0 interpolation based on temperature values from the stored R0 data, finally writes the result back to file."""

        data = self.read_input_data()

        def interpolate(temp):
            # Create result array with same shape as input
            result = np.full_like(temp, np.nan, dtype=float)

            # Find valid temperature values (equivalent to R's valid mask)
            valid_mask = (
                ~np.isnan(temp) & (temp >= self.min_temp) & (temp <= self.max_temp)
            )

            # Only interpolate where we have valid values
            if np.any(valid_mask):
                result[valid_mask] = np.interp(
                    temp[valid_mask],  # Only pass valid values
                    self.r0_data.Temperature.values,
                    self.r0_data.Median_R0.values,
                    left=np.nan,
                    right=np.nan,
                )

            return result

        # run synchronously on one cpu if thhe run mode is "forbidden", which comes in handy for small data
        if self.run_mode == "forbidden":
            r0_map = xr.DataArray(
                interpolate(data["t2m"].values),
                coords=data["t2m"].coords,
                dims=data["t2m"].dims,
                name="r0",
            )
        else:
            r0_map = xr.apply_ufunc(
                interpolate,
                data["t2m"],
                input_core_dims=[[]],
                output_core_dims=[[]],
                vectorize=True,
                dask="parallelized",
                keep_attrs=True,
            )

        self.store_output_data(r0_map)

    def store_output_data(self, data: oneData) -> None:
        """Stores the processed data to the specified output file.
        *for now, this is netcdf4, later we will have a database connection for this*

        Args:
            data (oneData): Dataset to store
        """
        data.to_netcdf(self.output)
