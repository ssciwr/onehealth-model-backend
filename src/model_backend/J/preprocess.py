from typing import Dict, Any
import logging
import cdsapi
import tempfile
import yaml
import numpy as np
import xarray as xr
from pathlib import Path

from ..base.types import oneData

# this is left here for compatibility and to avoid dependencies during early stage development
# throw out when preprocessing is fully implemented. Talk to Tuyen about this.


class JPreprocessor:
    """JPreprocessor handles preprocessing tasks specific to the model type 'JModel'.
    This consolidates the data preprocessing steps of the original R code into a Python class that can be used within the model backend.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the JPreprocessor with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the preprocessor.
        """
        if "Download" not in config:
            raise ValueError(
                "Configuration must contain 'Download' key with necessary parameters."
            )

        if "Process" not in config:
            raise ValueError(
                "Configuration must contain 'Process' key with necessary parameters."
            )

        super().__init__(preprocessor_name="JPreprocessor", config=config)

        # TODO: make sure this is cleaned up, not obvious when this directory is deleted again by the system
        if self.config["target_file"] is None:
            tgt = tempfile.mkdtemp()
            self.config["target"] = Path(tgt, "downloaded_data").absolute()
        else:
            self.config["target"] = Path(self.config["target_file"]).absolute()

    @classmethod
    def from_config(cls, config_path: str | Path):
        """Built a new instance of JPreprocessor from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters for the preprocessor.

        Returns:
            JPreprocessor: An instance of JPreprocessor initialized with the provided configuration.
        """
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        return cls(config=config)

    def download_data(self) -> None:
        """Downloads data  using the cdsapi library."""

        logging.info("Downloading data in JPreprocessor")

        if self.config["target_file"].exists():
            logging.debug("Target file already exists, skipping download.")
            return  # If the target file already exists, skip downloading

        client = cdsapi.Client()

        client.retrieve(
            self.config["dataset"],
            {
                "product_type": self.config["Download"]["product_type"],
                "variable": self.config["Download"]["variable"],
                "year": self.config["Download"]["year"],
                "month": self.config["Download"]["month"],
                "day": self.config["Download"]["day"],
                "time": self.config["Download"]["time"],
                "data_format": self.config["Download"]["format"],
                "download_format": self.config["Download"]["download_format"],
            },
            str(self.config["target"]),
        )

    def process_data(self, input: xr.Dataset) -> oneData:
        """Processes the input data for the JModel.
        This function processes the input data by adjusting the longitude range, converting temperature from Kelvin to Celsius, and saving the processed data if specified in the configuration.

        Args:
            input (xr.Dataset):  The input data/filepath/other things to be processed. This must be an xarray Dataset.

        Returns:
            oneData: Processed output data, which can be an xr.Dataset, xr.DataArray, or np.ndarray in general. Here, this is always an xr.Dataset.
        """
        # this function processes data. This is a copy of the supplied code, but it is not entirely clear how this should be generalized, and at any rate, it is only used here for as long as the standard preprocessing is not yet available.
        logging.info("Processing data in JPreprocessor")

        # Preserve original longitude attributes and adjust longitude range to [-180, 180]
        lon_attrs = input["longitude"].attrs
        input = input.assign_coords(
            longitude=((input.longitude + 180) % 360 - 180)
        ).sortby("longitude")
        input["longitude"].attrs = lon_attrs

        t2m_attrs = input["t2m"].attrs

        t2m_attrs.update(
            {
                "GRIB_longitudeOfFirstGridPointInDegrees": np.float64(
                    self.config["Process"]["longitude_of_first_grid_point"]
                ),
                "GRIB_longitudeOfLastGridPointInDegrees": np.float64(
                    self.config["Process"]["longitude_of_last_grid_point"]
                ),
                "GRIB_units": self.config["Process"]["temperature_units"],
                "units": self.config["Process"]["temperature_units"],
            }
        )

        # Convert temperature from Kelvin to Celsius
        input["t2m"] = input["t2m"] - self.config["Process"]["temperature_offset"]

        if self.config["Process"]["save_processed"]:
            input.to_netcdf(
                path=self.config["target_file"].with_stem(
                    self.config["target_file"].stem + "_processed"
                ),
                mode="w",
                encoding={
                    "t2m": {
                        "zlib": True,
                        "complevel": self.config["Process"]["compression_level"],
                    }
                },
            )

        return input

    def process(self, input: xr.Dataset) -> oneData:
        """Processes the input data for the JModel.

        Args:
            input (Any): The input data/filepath/other things to be processed. This can be any type.

        Returns:
            oneData: Processed output data, which can be an xr.Dataset, xr.DataArray, or np.ndarray.
        """
        # Implement the logic for preprocessing the input data here
        # For demonstration, we will just return a dummy xr.Dataset
        logging.info("Processing input data in JPreprocessor")

        self.download_data()

        data = xr.open_dataset(self.config["Download"]["target_file"]).load()

        if self.config["Process"]["process_data"]:
            return self.process_data(data)
        else:
            return data
