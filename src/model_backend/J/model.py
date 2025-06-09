from typing import Dict, Any
import xarray as xr
import numpy as np
import pandas as pd
from scipy import interp1d

from ..utils.read_data import read_geodata


type oneData = xr.Dataset | xr.DataArray | np.ndarray


# there is some potential for generalization here - all the dask stuff could go into a
# base class, and the individual models could just implement the model-specific logic
# and the creation of the task graph?
class JModel:
    """
    Implements the model used in the paper *TODO*
    """

    name: str = "JModel"
    task_graph: Dict[str, Any] | None = None
    input: str | None = None  # Placeholder for input data, to be defined later
    output: str | None = None  # Placeholder for output data, to be defined later
    run_mode: str = (
        "synchronous"  # Default run mode, can be changed based on configuration
    )
    r0_data: xr.Dataset | None = None  # Placeholder for R0 data, to be defined later
    min_temp: np.float64 = 0.0  # Minimum temperature for interpolation
    max_temp: np.float64 = 45.0  # Maximum temperature for interpolation
    step: np.float64 = 0.1  # Step size for temperature interpolation
    grid_data: xr.Dataset | None = (
        None  # Placeholder for grid data, to be defined later
    )
    nuts_level: int = 0  # NUTS level for the model, default is 0
    resolution: str = "10M"  # Resolution for the nuts data
    year: int = 2024
    month: int | None = None  # Month for the model, default is None

    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """Initializes the JModel with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """

        # set up plumbing for the model
        self.run_mode = config["run_mode"]

        if self.run_mode not in ["synchronous", "processes", "distributed", "threads"]:
            raise ValueError(
                f"Invalid run mode: {self.run_mode}. Supported modes are 'single', 'processes', 'distributed', 'threads'."
            )

        r0_path = config["r0_data"]
        self.r0_data = xr.Dataset(pd.read_csv(r0_path)) if r0_path else None
        self.output = config["output"]

        # get data variable to read, but do not read it yet
        self.input_file = config.get("input_file", None)
        self.data_format = config["data_format"]
        self.step_temp = (config.get("step", 0.1),)
        self.min_temp = (config.get("min_temp", 0.0),)
        self.max_temp = (config.get("max_temp", 45.0),) + self.step_temp

        grid_data_base_url = config.get("grid_data", None)
        self.nuts_level = config.get("nuts_level", 3)
        self.resolution = config.get("resolution", "10M")
        self.year = config.get("year", 2024)
        self.month = config.get("month", None)

        self.grid_data = read_geodata(
            base_url=grid_data_base_url,
            nuts_level=self.nuts_level,
            resolution=self.resolution,
            year=self.year,
            month=self.month,
            url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
        )

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "JModel":
        """Build a new instance of JModel from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters for the model.

        Returns:
            JModel: An instance of JModel initialized with the provided configuration.
        """
        return cls(config)

    def read_input_data(self) -> oneData:
        """Read input data from given source 'self.input'

        Returns:
            oneData: xarray dataset or data array containing the input data for the model.
        """

        # nothing done here yet
        data = xr.open_dataset(
            self.input_file, engine="netcdf4", decode_coords=True, chunks="auto"
        )

        if data is None:
            raise ValueError("Input data source is not defined in the configuration.")

        return data

    def run(
        self,
    ) -> None:
        """Runs the JModel with the provided input data.

        Args:
            input_data (xr.Dataset | xr.DataArray): Input data to be processed by the model.

        Returns:
            xr.Dataset|xr.DataArray: Processed output data from the model.
        """

        data = self.read_input_data()

        valid = (
            ~np.isnan(self.r0_data.Median_R0)
            & (self.r0_data.Temperature >= self.min_temp)
            & (self.r0_data.Temperature <= self.max_temp)
        ).values

        interp_function = interp1d(
            self.r0_data.Temperature[valid].values,
            self.r0_data.Median_R0[valid].values,
            bounds_error=False,
            fill_value=0.0,
            kind="linear",
        )

        # TODO: make sure this works as intended, current results differ from R
        r0_map = xr.Dataset(
            xr.apply_ufunc(
                interp_function,
                data["t2m"],
                dask="allowed",
                input_core_dims=[[]],
                output_core_dims=[[]],
                vectorize=True,
            )
        )

        self.store_output_data(r0_map)

    def store_output_data(self, data: oneData) -> None:
        """Stores the processed data to the specified output file.
        *for now, this is netcdf4, later we will have a database connection for this*

        Args:
            data (oneData): Dataset to store
        """
        xr.Dataset.to_netcdf(
            data,
            self.output,
            format="NETCDF4",
        )
