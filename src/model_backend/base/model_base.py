from abc import ABC, abstractmethod
from typing import Dict, Any

from .types import oneData


class ModelBase(ABC):
    def __init__(self, model_name: str, config: Dict[str:Any]):
        """Initializes the base model with a name and configuration.
        Args:
            model_name (str): Name of the model.
            config (Dict[str, Any]): Configuration dictionary for the model.
        """
        self.model_name = model_name
        self.config = config

    @property
    def name(self) -> str:
        """Returns the name of the model."""
        return self._model_name

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the configuration of the model."""
        return self._config

    @abstractmethod
    def run(
        self, input_data: oneData
    ) -> oneData:
        """Runs the model with the provided input data.
        This method should be implemented by subclasses to define how the model processes the input data.

        Args:
            input_data (xr.Dataset | xr.DataArray | np.ndarray): Input data to be processed by the model.

        Returns:
            xr.Dataset|xr.DataArray|np.ndarray: Processed output data from the model.
        """
        pass
