from ..base.model_base import BaseModel 

from typing import Dict, Any
import logging 
from ..base.types import oneData

class JModel(BaseModel):
    """Class that extends BaseModel to handle model-specific tasks for the model type 'JModel'.

    Args:
        BaseModel (AbstractBaseClass): Base class for models, providing a structure for model operations.
    """

    def __init__(self, config: Dict[str, Any], ):
        """Initializes the JModel with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        super().__init__(model_name="JModel", config=config)

    @classmethod 
    def from_config(cls, config: Dict[str, Any]) -> 'JModel':
        """Build a new instance of JModel from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters for the model.

        Returns:
            JModel: An instance of JModel initialized with the provided configuration.
        """
        return cls(config=config)
    
    def run(self, input_data: oneData) -> oneData:
        """Runs the JModel with the provided input data.

        Args:
            input_data (xr.Dataset | xr.DataArray): Input data to be processed by the model.

        Returns:
            xr.Dataset|xr.DataArray: Processed output data from the model.
        """
        # Implement the logic for running the model here
        # For demonstration, we will just return the input data
        return input_data