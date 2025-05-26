from ..base.processor_base import PreprocessorBase 

from typing import Dict, Any
import logging 
import numpy as np 
import xarray as xr

class JPreprocessor(PreprocessorBase):
    """JPreprocessor is a class that extends PreprocessorBase to handle preprocessing tasks specific to the model type 'JModel'.

    Args:
        PreprocessorBase (_type_): Base class for preprocessors, providing a structure for preprocessing tasks.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes the JPreprocessor with the given configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary for the preprocessor.
        """
        super().__init__(preprocessor_name="JPreprocessor", config=config)

    @classmethod
    def from_config(cls, config: Dict[str, Any]):
        """Built a new instance of JPreprocessor from a configuration dictionary.

        Args:
            config (Dict[str, Any]): Dictionary containing configuration parameters for the preprocessor.

        Returns:
            JPreprocessor: An instance of JPreprocessor initialized with the provided configuration.
        """
        return cls(config=config)
    
