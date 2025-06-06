from typing import Dict, Any
import dask
import xarray as xr
from ..base.types import oneData


# there is some potential for generalization here - all the dask stuff could go into a
# base class, and the individual models could just implement the model-specific logic
# and the creation of the task graph?
class JModel:
    """Class that extends BaseModel to handle model-specific tasks for the model type 'JModel'.

    Args:
        BaseModel (AbstractBaseClass): Base class for models, providing a structure for model operations.
    """

    name: str = "JModel"
    description: str = (
        "JModel handles model-specific tasks for the model type 'JModel'."
    )
    task_graph: Dict[str, Any] | None = None
    input: str | None = None  # Placeholder for input data, to be defined later
    output: str | None = None  # Placeholder for output data, to be defined later
    run_mode: str = (
        "synchronous"  # Default run mode, can be changed based on configuration
    )

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

        self.output = config.get("output", "JModel_output.nc")

        # get data variable to read, but do not read it yet
        self.input_file = config.get("data", None)
        self.data_format = config.get("data_format", "NETCDF4")

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

    def run(self) -> None:
        """Runs the JModel with the provided input data.

        Args:
            input_data (xr.Dataset | xr.DataArray): Input data to be processed by the model.

        Returns:
            xr.Dataset|xr.DataArray: Processed output data from the model.
        """
        data = self.read_input_data()

        graph = self.task_graph
        if graph is None:
            self.define_task_graph()

        graph["input"] = data
        result = dask.compute(self.computation, scheduler=self.run_mode)

        self.store_output_data(result[0])

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

    def define_task_graph(self) -> None:
        """Defines the task graph for the JModel."""

        transform_task = dask.delayed(self._transform_transmission_rates)
        interpolate_task = dask.delayed(self._interpolate_transmission_rates)

        input = dask.delayed(lambda x: x, pure=True)(xr.Dataset())
        transformed = transform_task(input)
        interpolated = interpolate_task(transformed)
        self.input_key = input.key
        self.computation = interpolated
        self.task_graph = dict(interpolated.dask)

    # individual functions that define the model logic
    def _interpolate_transmission_rates(self, data: oneData) -> oneData:
        return data

    def _transform_transmission_rates(self, data: oneData) -> oneData:
        return data
