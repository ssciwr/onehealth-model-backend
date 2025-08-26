from dataclasses import dataclass

import numpy as np
import xarray as xr


@dataclass
class PmodelInput:
    initial_conditions: np.ndarray
    latitude: xr.DataArray
    population_density: xr.DataArray
    rainfall: xr.DataArray
    temperature: xr.DataArray
    temperature_mean: xr.DataArray

    def print_attributes(self) -> None:
        """Prints all attributes of the class instance.

        This method dynamically lists all attributes in the instance,
        making it useful when the class is updated with new attributes.
        """
        print("PmodelInput Attributes:")

        # Get all instance variables using __annotations__ to include type hints
        attributes = self.__annotations__ if hasattr(self, "__annotations__") else {}

        # Add any instance attributes that might not be in annotations
        for attr_name in dir(self):
            # Filter out methods, private attributes, and special methods
            if (
                not attr_name.startswith("_")
                and not callable(getattr(self, attr_name))
                and attr_name not in attributes
            ):
                attributes[attr_name] = type(getattr(self, attr_name)).__name__

        # Print each attribute with its type and value information
        for attr_name, attr_type in attributes.items():
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if attr_value is None:
                    print(f"  - {attr_name}: None")
                elif hasattr(attr_value, "shape"):
                    print(
                        f"  - {attr_name}: {type(attr_value).__name__} with shape {attr_value.shape}"
                    )
                else:
                    print(f"  - {attr_name}: {type(attr_value).__name__}")
            else:
                print(f"  - {attr_name}: Not set")
