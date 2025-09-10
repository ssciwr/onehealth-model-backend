import pytest
import numpy as np

from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_mort_e
from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_mort_j
from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_mort_a
from heiplanet_models.Pmodel.Pmodel_rates_mortality import mosq_surv_ed


# ---- Pytest Fixtures
@pytest.fixture
def typical_temperature_array():
    """Provides a 1D numpy array of typical temperature values."""
    return np.array([10, 20, 25, 30], dtype=float)


@pytest.fixture
def edge_case_temperature_array():
    """Provides an array of edge case temperature values."""
    return np.array([0, 50, -10, 100, 1000], dtype=float)


@pytest.fixture
def negative_temperature_array():
    """Provides an array of negative temperature values."""
    return np.array([-10, 0, 10], dtype=float)


@pytest.fixture
def empty_temperature_array():
    """Provides an empty numpy array."""
    return np.array([])


@pytest.fixture
def multidimensional_temperature_array():
    """Provides a 2D numpy array of typical temperature values."""
    return np.array([[10, 20], [25, 30]], dtype=float)


@pytest.fixture
def typical_3d_temperature_array():
    """Provides a 3D numpy array of typical temperature values."""
    return np.array(
        [[[10, 20, 15], [25, 18, 22]], [[5, 12, 8], [15, 10, 14]]], dtype=float
    )


@pytest.fixture
def edge_case_3d_temperature_array():
    """Provides a 3D numpy array of edge case temperature values."""
    return np.array(
        [[[0, 50, -10], [100, 1000, -5]], [[-20, 0, 10], [40, 60, -1]]], dtype=float
    )


@pytest.fixture
def negative_3d_temperature_array():
    """Provides a 3D numpy array of negative temperature values."""
    return np.array(
        [[[-10, -5, -1], [-15, -20, -2]], [[-8, -12, -3], [-7, -14, -4]]], dtype=float
    )


@pytest.fixture
def large_3d_temperature_array():
    """Provides a 3D numpy array of large temperature values."""
    return np.array(
        [[[100, 1000, 150], [200, 500, 300]], [[80, 120, 110], [90, 140, 130]]],
        dtype=float,
    )


@pytest.fixture
def empty_3d_temperature_array():
    """Provides an empty 3D numpy array."""
    return np.empty((2, 2, 0), dtype=float)


@pytest.fixture
def temperature_array_4x3x2():
    """Provides a 3D numpy array with dimensions (4, 3, 2)."""
    return np.array(
        [
            [[10, 20], [15, 25], [12, 22]],
            [[8, 18], [13, 23], [10, 20]],
            [[5, 15], [10, 20], [8, 18]],
            [[7, 17], [12, 22], [9, 19]],
        ],
        dtype=float,
    )


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


# ---- Unit Tests for mosq_mort_a
def test_mosq_mort_a_typical_temperatures(typical_temperature_array):
    """Test mosq_mort_a with typical temperature values."""
    result = mosq_mort_a(typical_temperature_array)
    assert result.shape == typical_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)


def test_mosq_mort_a_scalar_input():
    """Test mosq_mort_a with a scalar temperature value."""
    temp = 25.0
    result = mosq_mort_a(temp)
    assert np.isscalar(result) or isinstance(result, float)
    assert np.isfinite(result)


def test_mosq_mort_a_multidimensional_input(multidimensional_temperature_array):
    """Test mosq_mort_a with multi-dimensional input arrays."""
    result = mosq_mort_a(multidimensional_temperature_array)
    assert result.shape == multidimensional_temperature_array.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_a_edge_case_temperatures(edge_case_temperature_array):
    """Test mosq_mort_a with edge case temperatures."""
    result = mosq_mort_a(edge_case_temperature_array)
    assert result.shape == edge_case_temperature_array.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_mort_a_negative_temperatures(negative_temperature_array):
    """Test mosq_mort_a with negative temperature values."""
    result = mosq_mort_a(negative_temperature_array)
    assert result.shape == negative_temperature_array.shape
    assert np.all(np.isfinite(result))


def test_mosq_mort_a_large_temperatures():
    """Test mosq_mort_a with very large temperature values."""
    temps = np.array([100, 1000], dtype=float)
    result = mosq_mort_a(temps)
    assert result.shape == temps.shape
    assert np.all(np.isfinite(result) | np.isinf(result))


def test_mosq_mort_a_empty_array(empty_temperature_array):
    """Test mosq_mort_a with an empty array."""
    result = mosq_mort_a(empty_temperature_array)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_mort_a_non_numeric_input():
    """Test mosq_mort_a with non-numeric input values."""
    with pytest.raises(TypeError):
        mosq_mort_a(np.array(["a", "b"], dtype=object))


def test_mosq_mort_a_output_consistency():
    """Test mosq_mort_a output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MORTALITY_MOSQUITO_A

    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_5"]
    CONST_6 = CONSTANTS_MORTALITY_MOSQUITO_A["CONST_6"]
    T = np.array([15.0, 25.0])
    expected = -np.log(
        CONST_1 * np.exp(CONST_2 * ((T - CONST_3) / CONST_4) ** CONST_5) * T**CONST_6
    )
    result = mosq_mort_a(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_mort_a_known_matrix_output():
    """Test mosq_mort_a returns expected output for a specific matrix."""
    T = np.array([[15.0, 20.0], [25.0, 30.0]])
    expected = np.array([[0.123266, 0.090511], [0.068645, 0.103640]])
    result = mosq_mort_a(T)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_mosq_mort_a_numerical_stability_near_zero():
    """Test mosq_mort_a avoids log(0) errors for near-zero intermediate results."""
    T = np.array([-1e6, 1e6], dtype=float)
    result = mosq_mort_a(T)
    assert np.all(np.isfinite(result) | np.isinf(result))


# ---- Unit Tests for mosq_surv_ed
def test_mosq_surv_ed_typical_temperatures(typical_3d_temperature_array):
    """Test mosq_surv_ed with typical temperature values."""
    result = mosq_surv_ed(typical_3d_temperature_array)
    assert result.shape == typical_3d_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert isinstance(result, np.ndarray)
    assert np.all((result >= 0) & (result <= 1))


def test_mosq_surv_ed_scalar_input():
    """Test mosq_surv_ed with a scalar temperature value."""
    temp = 25.0
    with pytest.raises(ValueError):
        mosq_surv_ed(temp)


def test_mosq_surv_ed_edge_case_temperatures(edge_case_3d_temperature_array):
    """Test mosq_surv_ed with edge case temperatures."""
    result = mosq_surv_ed(edge_case_3d_temperature_array)
    assert result.shape == edge_case_3d_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


def test_mosq_surv_ed_negative_temperatures(negative_3d_temperature_array):
    """Test mosq_surv_ed with negative temperature values."""
    result = mosq_surv_ed(negative_3d_temperature_array)
    assert result.shape == negative_3d_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


def test_mosq_surv_ed_large_temperatures(large_3d_temperature_array):
    """Test mosq_surv_ed with very large temperature values."""
    result = mosq_surv_ed(large_3d_temperature_array)
    assert result.shape == large_3d_temperature_array.shape
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


def test_mosq_surv_ed_empty_array(empty_3d_temperature_array):
    """Test mosq_surv_ed with an empty array."""
    result = mosq_surv_ed(empty_3d_temperature_array)
    assert result.size == 0
    assert isinstance(result, np.ndarray)


def test_mosq_surv_ed_non_numeric_input():
    """Test mosq_surv_ed with non-numeric input values."""
    with pytest.raises(TypeError):
        mosq_surv_ed(np.array([[[1, "a"]]], dtype=object))


def test_mosq_surv_ed_output_consistency():
    """Test mosq_surv_ed output matches manual calculation for known input."""
    from heiplanet_models.Pmodel.Pmodel_params import CONSTANTS_MORTALITY_MOSQUITO_ED

    ED_SURV_BL = CONSTANTS_MORTALITY_MOSQUITO_ED["ED_SURV_BL"]
    CONST_1 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_1"]
    CONST_2 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_2"]
    CONST_3 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_3"]
    CONST_4 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_4"]
    CONST_5 = CONSTANTS_MORTALITY_MOSQUITO_ED["CONST_5"]

    T = np.array([[[15.0, 10.0], [25.0, 20.0]]])
    T_min = T.copy()
    for k in range(1, T.shape[2]):
        T_min[:, :, k] = np.minimum(T_min[:, :, k - 1], T_min[:, :, k])

    expected = (
        ED_SURV_BL
        * CONST_1
        * np.exp(CONST_2 * ((T_min - CONST_3) / CONST_4) ** CONST_5)
    )
    result = mosq_surv_ed(T)
    np.testing.assert_allclose(result, expected)


def test_mosq_surv_ed_multidimensional_input_2d(multidimensional_temperature_array):
    """Test mosq_surv_ed with 2D input, expecting failure."""
    with pytest.raises(ValueError):
        mosq_surv_ed(multidimensional_temperature_array)


def test_mosq_surv_ed_known_matrix_output(temperature_array_4x3x2):
    """Test mosq_surv_ed returns expected output for a specific matrix."""
    # Example expected values; update as needed for your constants
    T = temperature_array_4x3x2
    expected = np.array(
        [
            [
                [0.9299, 0.9299],
                [0.9299, 0.9299],
                [0.9300, 0.9300],
            ],
            [[0.9299, 0.9299], [0.9299, 0.9299], [0.9299, 0.9299]],
            [[0.9272, 0.9272], [0.9299, 0.9299], [0.9299, 0.9299]],
            [[0.9296, 0.9296], [0.9300, 0.9300], [0.9299, 0.9299]],
        ]
    )

    result = mosq_surv_ed(T)
    print(result)
    np.testing.assert_allclose(result, expected, atol=1e-4)
