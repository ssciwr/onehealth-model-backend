from abc import ABC, abstractmethod
from typing import Dict, Any
from ..base.types import oneData
class PreprocessorBase(ABC):
    def __init__(self, preprocessor_name: str, config: Dict[str, Any]):
        """_summary_

        Args:
            preprocessor_name (str): _description_
            config (Dict[str, Any]): _description_
        """
        self.preprocessor_name = preprocessor_name
        self.config = config

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        return self.preprocessor_name

    @property
    def config(self) -> Dict[str, Any]:
        """_summary_

        Returns:
            Dict[str, Any]: _description_
        """
        return self._config

    @abstractmethod
    def process(self, input: Any) -> oneData:
        """        Abstract method to process the input data.
        This method should be implemented by subclasses to define how the input data is processed.

        Args:
            input (Any): The input data/filepath/other things to be processed. This can be any type.
        """
        pass