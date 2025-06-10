import numpy as np
from scipy.interpolate import make_interp_spline
from typing import Callable, Union, Dict, Type


# Base interface (optional, for clarity)
class InterpolatorBase:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self._validate_data()

    def _validate_data(self):
        if self.x_data.ndim != 1 or self.y_data.ndim != 1:
            raise ValueError("x_data and y_data must be 1-dimensional arrays.")
        if len(self.x_data) != len(self.y_data):
            raise ValueError("x_data and y_data must have the same length.")
        if len(self.x_data) < 2:
            raise ValueError("At least two data points are required.")
        if not np.all(np.diff(self.x_data) >= 0):
            raise ValueError("x_data must be sorted in ascending order.")

    def __call__(self, x_new: Union[float, np.ndarray]) -> np.ndarray:
        raise NotImplementedError("Interpolator must implement __call__ method.")


class LinearInterpolator(InterpolatorBase):
    def __call__(self, x_new: Union[float, np.ndarray]) -> np.ndarray:
        return np.interp(x_new, self.x_data, self.y_data)


class BSplineInterpolator(InterpolatorBase):
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, degree: int = 3):
        super().__init__(x_data, y_data)
        if not isinstance(degree, int) or not (1 <= degree <= 5):
            raise ValueError("degree must be an integer between 1 and 5.")
        self.degree = degree
        self.spline = make_interp_spline(self.x_data, self.y_data, k=self.degree)

    def __call__(self, x_new: Union[float, np.ndarray]) -> np.ndarray:
        return self.spline(x_new)


def interpolator_function(
    method: str, x_data: np.ndarray, y_data: np.ndarray, **kwargs
) -> Callable[[Union[float, np.ndarray]], np.ndarray]:
    """
    Factory function to create an interpolator.

    Parameters:
    -----------
    method : str
        Interpolation method, e.g., 'linear' or 'bspline'.
    x_data : array-like
        1D array of x data points (must be sorted ascending).
    y_data : array-like
        1D array of y data points.
    kwargs:
        Additional parameters for specific interpolators,
        e.g. degree for bspline.

    Returns:
    --------
    Callable interpolator function.
    """

    # Registry to hold interpolators
    _INTERPOLATORS: Dict[str, Type[InterpolatorBase]] = {
        "linear": LinearInterpolator,
        "bspline": BSplineInterpolator,
    }

    method = method.lower()
    cls = _INTERPOLATORS.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown interpolation method '{method}'. Supported methods: {list(_INTERPOLATORS.keys())}"
        )

    return cls(x_data, y_data, **kwargs)
