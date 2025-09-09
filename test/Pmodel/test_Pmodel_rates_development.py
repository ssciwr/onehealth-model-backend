import pytest

import numpy as np

from heiplanet_models.Pmodel.Pmodel_rates_development import mosq_dev_j


# ---- Pytest Fixtures
@pytest.fixture
def typical_temperature_array():
    """Provides a 1D numpy array of typical temperature values."""
    return np.array([10, 20, 25, 30], dtype=float)


# ---- Unit Tests for mosq_dev_j
def test_mosq_dev_j_typical_temperatures(typical_temperature_array):
    """Test mosq_dev_j with typical temperature values."""
    result = mosq_dev_j(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_dev_j_scalar_input():
    """Test mosq_dev_j with a scalar temperature value."""
    temp = 25.0
    result = mosq_dev_j(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_dev_j_edge_case_temperatures():
    """Test mosq_dev_j with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_j_negative_temperatures():
    """Test mosq_dev_j with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_j_large_temperatures():
    """Test mosq_dev_j with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_dev_j_empty_array():
    """Test mosq_dev_j with an empty array."""
    temps = np.array([])
    result = mosq_dev_j(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_dev_j_non_numeric_input():
    """Test mosq_dev_j with non-numeric input values."""
    with pytest.raises((TypeError, ValueError)):
        mosq_dev_j(np.array(["a", None, np.nan], dtype=object))


def test_mosq_dev_j_output_consistency():
    """Test mosq_dev_j output matches manual calculation for known input."""
    # Example constants (should match those in the model)
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_J

    CONST_1 = CONSTANTS_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_J["CONST_4"]
    T = np.array([15.0, 25.0])
    expected = CONST_4 / (CONST_1 - CONST_2 * T + CONST_3 * T**2)
    result = mosq_dev_j(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_dev_j_multidimensional_input():
    """Test mosq_dev_j with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_dev_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))
