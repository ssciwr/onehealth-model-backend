import pytest

import numpy as np

from heiplanet_models.Pmodel.Pmodel_rates_development import mosq_dev_j, mosq_dev_i


# ---- Pytest Fixtures
@pytest.fixture
def typical_temperature_array():
    """Provides a 1D numpy array of typical temperature values."""
    return np.array([10, 20, 25, 30], dtype=float)


@pytest.fixture
def typical_temperature_array_i():
    """Provides a 1D numpy array of typical temperature values for mosq_dev_i."""
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


def test_mosq_dev_j_known_matrix_output():
    """Test mosq_dev_j returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.036536, 0.058754], [0.093720, 0.120192]])
    result = mosq_dev_j(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---- Unit Tests for mosq_dev_i
def test_mosq_dev_i_typical_temperatures(typical_temperature_array_i):
    """Test mosq_dev_i with typical temperature values."""
    result = mosq_dev_i(typical_temperature_array_i)
    assert result.shape == typical_temperature_array_i.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_dev_i_scalar_input():
    """Test mosq_dev_i with a scalar temperature value."""
    temp = 25.0
    result = mosq_dev_i(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_dev_i_edge_case_temperatures():
    """Test mosq_dev_i with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_negative_temperatures():
    """Test mosq_dev_i with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_large_temperatures():
    """Test mosq_dev_i with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_dev_i_non_numeric_input():
    """Test mosq_dev_i with non-numeric input values."""
    with pytest.raises((TypeError, ValueError)):
        mosq_dev_i(np.array(["a", None, np.nan], dtype=object))


def test_mosq_dev_i_output_consistency():
    """Test mosq_dev_i output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MOSQUITO_I

    CONST_1 = CONSTANTS_MOSQUITO_I["CONST_1"]
    CONST_2 = CONSTANTS_MOSQUITO_I["CONST_2"]
    CONST_3 = CONSTANTS_MOSQUITO_I["CONST_3"]
    CONST_4 = CONSTANTS_MOSQUITO_I["CONST_4"]
    T = np.array([15.0, 25.0])
    expected = CONST_4 / (CONST_1 - CONST_2 * T + CONST_3 * T**2)
    result = mosq_dev_i(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_dev_i_multidimensional_input():
    """Test mosq_dev_i with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_dev_i(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_dev_i_empty_array():
    """Test mosq_dev_i with an empty array."""
    temps = np.array([])
    result = mosq_dev_i(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_dev_i_known_matrix_output():
    """Test mosq_dev_i returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.083229, 0.160772], [0.258065, 0.200803]])
    result = mosq_dev_i(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)
