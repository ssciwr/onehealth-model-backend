import pytest
import numpy as np

from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_mort_e
from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_mort_j


# ---- Pytest Fixtures
@pytest.fixture
def typical_temperature_array():
    """Provides a 1D numpy array of typical temperature values."""
    return np.array([10, 20, 25, 30], dtype=float)


# ---- Unit Tests for mosq_mort_e
def test_mosq_mort_e_typical_temperatures(typical_temperature_array):
    """Test mosq_mort_e with typical temperature values."""
    result = mosq_mort_e(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_mort_e_scalar_input():
    """Test mosq_mort_e with a scalar temperature value."""
    temp = 25.0
    result = mosq_mort_e(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_mort_e_edge_case_temperatures():
    """Test mosq_mort_e with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_mort_e(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_e_negative_temperatures():
    """Test mosq_mort_e with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_mort_e(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_e_large_temperatures():
    """Test mosq_mort_e with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_mort_e(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_mort_e_empty_array():
    """Test mosq_mort_e with an empty array."""
    temps = np.array([])
    result = mosq_mort_e(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_mort_e_non_numeric_input():
    """Test mosq_mort_e with non-numeric input values."""
    with pytest.raises(TypeError):
        mosq_mort_e(np.array(["a", "b"], dtype=object))


def test_mosq_mort_e_output_consistency():
    """Test mosq_mort_e output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MORTALITY_MOSQUITO_E

    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_E["CONST_5"]
    T = np.array([15.0, 25.0])
    expected = -np.log(CONST_1 * np.exp(CONST_2 * ((T - CONST_3) / CONST_4) ** CONST_5))
    result = mosq_mort_e(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_mort_e_multidimensional_input():
    """Test mosq_mort_e with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_mort_e(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_e_known_matrix_output():
    """Test mosq_mort_e returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.046059, 0.046044], [0.046329, 0.055953]])
    result = mosq_mort_e(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---- Unit Tests for mosq_mort_j
def test_mosq_mort_j_typical_temperatures(typical_temperature_array):
    """Test mosq_mort_j with typical temperature values."""
    result = mosq_mort_j(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_mort_j_scalar_input():
    """Test mosq_mort_j with a scalar temperature value."""
    temp = 25.0
    result = mosq_mort_j(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_mort_j_edge_case_temperatures():
    """Test mosq_mort_j with edge case temperatures."""
    temps = np.array([0, 50], dtype=float)
    result = mosq_mort_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_j_negative_temperatures():
    """Test mosq_mort_j with negative temperature values."""
    temps = np.array([-10, 0, 10], dtype=float)
    result = mosq_mort_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_j_large_temperatures():
    """Test mosq_mort_j with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_mort_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_mort_j_empty_array():
    """Test mosq_mort_j with an empty array."""
    temps = np.array([])
    result = mosq_mort_j(temps)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_mort_j_non_numeric_input():
    """Test mosq_mort_j with non-numeric input values."""
    with pytest.raises(TypeError):
        mosq_mort_j(np.array(["a", "b"], dtype=object))


def test_mosq_mort_j_output_consistency():
    """Test mosq_mort_j output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MORTALITY_MOSQUITO_J

    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_J["CONST_5"]
    T = np.array([15.0, 25.0])
    expected = -np.log(CONST_1 * np.exp(CONST_2 * ((T - CONST_3) / CONST_4) ** CONST_5))
    result = mosq_mort_j(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_mort_j_multidimensional_input():
    """Test mosq_mort_j with multi-dimensional input arrays."""
    temps = np.array([[20, 25], [30, 35]], dtype=float)
    result = mosq_mort_j(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_j_known_matrix_output():
    """Test mosq_mort_j returns expected output for a specific matrix."""
    # Example expected values; update as needed for your constants
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.025631, 0.023269], [0.023294, 0.030533]])
    result = mosq_mort_j(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_mosq_mort_j_numerical_stability_near_zero():
    """Test mosq_mort_j avoids log(0) errors for near-zero intermediate results."""
    # Create input that would result in a near-zero intermediate value
    T = np.array([-1e6, 1e6], dtype=float)
    result = mosq_mort_j(T)
    assert np.all(np.isfinite(result) | np.isinf(result))
