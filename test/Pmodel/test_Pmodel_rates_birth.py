import pytest
import math
from pathlib import Path

import numpy as np
import xarray as xr


from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    revolution_angle,
    declination_angle,
    daylight_forsythe,
    mosq_birth,
    mosq_dia_hatch,
    mosq_dia_lay,
)
from heiplanet_models.Pmodel.Pmodel_params import (
    CONSTANT_DECLINATION_ANGLE,
    CONSTANTS_MOSQUITO_BIRTH,
    CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING,
    CONSTANTS_MOSQUITO_DIAPAUSE_LAY,
    DAYS_YEAR,
    HALF_DAYS_YEAR,
)


# ---- Pytest Fixtures
@pytest.fixture
def mock_lay_data():
    """Provides mock xarray data for diapause lay tests for a single year."""
    lons = [10.0]
    lats = [45.0]  # Mid-latitude
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_multi_year():
    """Provides mock xarray data for diapause lay tests for two years."""
    lons = [10.0]
    lats = [45.0]
    times = np.arange(DAYS_YEAR * 2)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_equator():
    """Provides mock xarray data for diapause lay tests at the equator."""
    lons = [10.0]
    lats = [0.0]  # Equator
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_lay_data_polar():
    """Provides mock xarray data for diapause lay tests at a polar latitude."""
    lons = [10.0]
    lats = [80.0]  # Polar latitude
    times = np.arange(DAYS_YEAR)
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data():
    """Provides mock xarray data for diapause hatching tests."""
    lons = [10.0, 11.0]
    lats = [45.0, 46.0]
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    temp_data = xr.DataArray(
        np.zeros((len(lons), len(lats), len(times))), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data_below_ctt():
    """Provides mock xarray data for diapause hatching tests with temperatures below CTT."""
    lons = [10.0, 11.0]
    lats = [45.0, 46.0]
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    # All temperatures set below CTT
    temp_data = xr.DataArray(
        np.full((len(lons), len(lats), len(times)), CTT - 5), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def mock_hatch_data_high_latitude():
    """Provides mock xarray data for diapause hatching tests at high latitude."""
    lons = [10.0, 11.0]
    lats = [80.0]  # High latitude for short daylight
    times = np.arange(60, dtype=int)  # 60 days
    coords = {"longitude": lons, "latitude": lats, "time": times}
    dims = ("longitude", "latitude", "time")

    CTT = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING["CTT"]
    # Temperatures always above CTT
    temp_data = xr.DataArray(
        np.full((len(lons), len(lats), len(times)), CTT + 10), dims=dims, coords=coords
    )
    lat_data = xr.DataArray(lats, dims=("latitude"), coords={"latitude": lats})
    return temp_data, lat_data


@pytest.fixture
def resources_path():
    """Provides the correct, absolute path to the test resources directory."""
    # The test file is in '.../test/Pmodel/'.
    # The resources are in '.../test/resources/'.
    # Path(__file__).parent gives the directory of the current test file ('.../test/Pmodel').
    # .parent then goes up one level to '.../test/'.
    # Finally, we join it with the 'resources' directory name.
    return Path(__file__).parent.parent / "resources"


@pytest.fixture
def temp_dummy_data(resources_path):
    """Loads the dummy temperature data from a NetCDF file."""
    return xr.open_dataarray(resources_path / "temperature_dummy.nc")


# ---- revolution_angle()
def test_revolution_angle_day_1():
    result = revolution_angle(1)
    assert isinstance(result, float)
    assert math.isclose(3.3161, result, rel_tol=1e-04, abs_tol=0.0)


def test_revolution_angle_day_365():
    result = revolution_angle(365)
    assert isinstance(result, float)
    assert math.isclose(3.2930, result, rel_tol=1e-04, abs_tol=0.0)


def test_revolution_angle_day_366():
    result = revolution_angle(366)
    assert isinstance(result, float)
    assert math.isclose(3.3108, result, rel_tol=1e-04, abs_tol=0.0)


def test_revolution_angle_mid_year():
    result = revolution_angle(180)
    assert isinstance(result, float)
    assert math.isclose(0.1165, result, rel_tol=1e-04, abs_tol=0.0)


def test_revolution_angle_day_zero_raises():
    with pytest.raises(ValueError):
        revolution_angle(0)


def test_revolution_angle_day_367_raises():
    with pytest.raises(ValueError):
        revolution_angle(367)


def test_revolution_angle_negative_day_raises():
    with pytest.raises(ValueError):
        revolution_angle(-5)


def test_revolution_angle_float_input_raises():
    with pytest.raises(TypeError):
        revolution_angle(1.5)


def test_revolution_angle_string_input_raises():
    with pytest.raises(TypeError):
        revolution_angle("100")


def test_revolution_angle_none_input_raises():
    with pytest.raises(TypeError):
        revolution_angle(None)


def test_revolution_angle_vector_input():
    """Test that the function correctly processes a vector of days."""
    days = np.array([1, 183, 365])
    results = revolution_angle(days)
    assert isinstance(results, np.ndarray)
    assert results.shape == (3,)
    # Check if the results for individual days match
    np.testing.assert_allclose(results[0], revolution_angle(1))
    np.testing.assert_allclose(results[1], revolution_angle(183))
    np.testing.assert_allclose(results[2], revolution_angle(365))


def test_revolution_angle_vector_with_invalid_day_raises():
    """Test that a vector with an out-of-range day raises a ValueError."""
    days = np.array([1, 367, 365])
    with pytest.raises(ValueError, match="All 'days' must be in the range 1 to 366."):
        revolution_angle(days)


def test_revolution_angle_vector_with_invalid_type_raises():
    """Test that a vector with a non-integer type raises a TypeError."""
    days = np.array([1.0, 183.5, 365.0])
    with pytest.raises(TypeError, match="Input 'days' must be an integer"):
        revolution_angle(days)


# ---- declination_angle()
def test_declination_angle_at_equinox():
    """Test declination angle is near zero at an equinox revolution angle."""
    expected = 2.41367e-17
    result = declination_angle(np.pi / 2)
    assert isinstance(result, float)
    assert math.isclose(result, expected, abs_tol=1e-4)


def test_declination_angle_at_solstice_positive():
    """Test declination at a solstice (revolution angle 0), should be max positive."""
    expected = np.arcsin(CONSTANT_DECLINATION_ANGLE)
    result = declination_angle(0.0)
    assert isinstance(result, float)
    assert math.isclose(result, expected, rel_tol=1e-4)


def test_declination_angle_at_solstice_negative():
    """Test declination at the other solstice (revolution angle PI), should be max negative."""
    expected = np.arcsin(-CONSTANT_DECLINATION_ANGLE)
    result = declination_angle(np.pi)
    assert isinstance(result, float)
    assert math.isclose(result, expected, rel_tol=1e-9)


def test_declination_angle_output_range():
    """Test that the output is within the valid range of arcsin [-PI/2, PI/2]."""
    result = declination_angle(1.5)  # Arbitrary angle
    assert -np.pi / 2 <= result <= np.pi / 2


def test_declination_angle_string_input_raises():
    with pytest.raises(TypeError):
        declination_angle("invalid")


def test_declination_angle_none_input_raises():
    with pytest.raises(TypeError):
        declination_angle(None)


# ---- daylight_forsythe()
def test_daylight_forsythe_equator_equinox():
    """Test daylight at the equator during an equinox, should be 12 hours."""
    result = daylight_forsythe(
        latitude=0.0, declination_angle=0.0, daylight_coefficient=0.0
    )
    assert isinstance(result, float)
    assert math.isclose(result, 12.0, abs_tol=1e-9)


def test_daylight_forsythe_polar_day():
    """Test daylight at North Pole during summer solstice, should be 24 hours."""
    # Use a latitude very close to 90 to avoid division by zero in the formula
    result = daylight_forsythe(
        latitude=89.999, declination_angle=0.409, daylight_coefficient=0.0
    )
    assert isinstance(result, float)
    assert math.isclose(result, 24.0, abs_tol=1e-4)


def test_daylight_forsythe_polar_night():
    """Test daylight at North Pole during winter solstice, should be 0 hours."""
    # Use a latitude very close to 90 to avoid division by zero
    result = daylight_forsythe(
        latitude=89.999, declination_angle=-0.409, daylight_coefficient=0.0
    )
    print(result)
    assert isinstance(result, float)
    assert math.isclose(result, 0.0, abs_tol=1e-4)


def test_daylight_forsythe_mid_latitude():
    """Test daylight at a mid-latitude during summer, should be > 12 hours."""
    result = daylight_forsythe(
        latitude=45.0, declination_angle=0.409, daylight_coefficient=0.0
    )
    assert isinstance(result, float)
    assert result > 12.0


def test_daylight_forsythe_invalid_latitude_high_raises():
    with pytest.raises(
        ValueError, match="Latitude must be between -90 and 90 degrees."
    ):
        daylight_forsythe(
            latitude=90.1, declination_angle=0.0, daylight_coefficient=0.0
        )


def test_daylight_forsythe_invalid_latitude_low_raises():
    with pytest.raises(
        ValueError, match="Latitude must be between -90 and 90 degrees."
    ):
        daylight_forsythe(
            latitude=-90.1, declination_angle=0.0, daylight_coefficient=0.0
        )


def test_daylight_forsythe_string_input_raises():
    with pytest.raises(TypeError):
        daylight_forsythe(
            latitude="invalid", declination_angle=0.0, daylight_coefficient=0.0
        )


def test_daylight_forsythe_none_input_raises():
    with pytest.raises(TypeError):
        daylight_forsythe(
            latitude=None, declination_angle=0.0, daylight_coefficient=0.0
        )


# ---- mosq_birth()
def test_mosq_birth_temp_above_threshold():
    """Test birth rate is zero when temperature is above the threshold."""
    const_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    temperature = np.array([const_1 + 1, const_1 + 10, 40])
    expected = np.array([0.0, 0.0, 0.0])
    result = mosq_birth(temperature)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_mosq_birth_temp_at_threshold():
    """Test birth rate is zero when temperature is exactly at the threshold."""
    const_1 = CONSTANTS_MOSQUITO_BIRTH["CONST_1"]
    temperature = np.array([const_1])
    expected = np.array([0.0])
    result = mosq_birth(temperature)
    np.testing.assert_allclose(result, expected, atol=1e-9)


def test_mosq_birth_temp_below_threshold():
    """Test birth rate is calculated correctly for a temperature below the threshold."""
    C = CONSTANTS_MOSQUITO_BIRTH
    temp = 20.0  # A value known to be below the typical threshold
    temperature = np.array([temp])
    # Manual calculation based on the formula in the function
    expected_value = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp) ** C["CONST_6"]
    )
    expected = np.array([expected_value])
    result = mosq_birth(temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_mosq_birth_mixed_temperatures():
    """Test birth rate with a mix of temperatures above, at, and below the threshold."""
    C = CONSTANTS_MOSQUITO_BIRTH
    const_1 = C["CONST_1"]
    temp_below = 15.0
    temperature = np.array([temp_below, const_1, const_1 + 5])

    expected_below = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp_below - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp_below) ** C["CONST_6"]
    )
    expected = np.array([expected_below, 0.0, 0.0])
    result = mosq_birth(temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_mosq_birth_zero_temperature():
    """Test birth rate calculation when the temperature is zero."""
    C = CONSTANTS_MOSQUITO_BIRTH
    temp = 0.0
    temperature = np.array([temp])
    expected_value = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp) ** C["CONST_6"]
    )
    expected = np.array([expected_value])
    result = mosq_birth(temperature)
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_mosq_birth_multidimensional_input():
    """Test that the function handles multi-dimensional arrays correctly."""
    C = CONSTANTS_MOSQUITO_BIRTH
    const_1 = C["CONST_1"]
    temp_below_a = 25.0
    temp_below_b = -5.0
    temperature = np.array([[temp_below_a, const_1 + 10], [const_1, temp_below_b]])

    expected_below_1 = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * ((temp_below_a - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - temp_below_a) ** C["CONST_6"]
    )
    expected_below_2 = (
        C["CONST_2"]
        * np.exp(C["CONST_3"] * (((temp_below_b) - C["CONST_4"]) / C["CONST_5"]) ** 2)
        * (C["CONST_1"] - (-5.0)) ** C["CONST_6"]
    )

    expected = np.array([[expected_below_1, 0.0], [0.0, expected_below_2]])
    result = mosq_birth(temperature)

    assert result.shape == temperature.shape
    np.testing.assert_allclose(result, expected, rtol=1e-7)


def test_mosq_birth_empty_input():
    """Test that the function handles an empty array input gracefully."""
    temperature = np.array([])
    expected = np.array([])
    result = mosq_birth(temperature)
    assert result.shape == expected.shape


# ---- mosq_dia_hatch()
def test_mosq_dia_hatch_invalid_temp_dims_raises(mock_hatch_data):
    """Test that a non-3D temperature array raises a ValueError."""
    _, lat_data = mock_hatch_data
    invalid_temp = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Temperature array must be 3D"):
        mosq_dia_hatch(invalid_temp, lat_data)


def test_mosq_dia_hatch_invalid_lat_dims_raises(mock_hatch_data):
    """Test that a non-1D latitude array raises a ValueError."""
    temp_data, _ = mock_hatch_data
    invalid_lat = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Latitude array must be 1D."):
        mosq_dia_hatch(temp_data, invalid_lat)


def test_mosq_dia_hatch_temp_below_ctt(mock_hatch_data_below_ctt):
    """Test that hatching is zero if temperature is always below the critical threshold."""
    temp_data, lat_data = mock_hatch_data_below_ctt
    result = mosq_dia_hatch(temp_data, lat_data)
    assert np.all(result.values == 0)


def test_mosq_dia_hatch_daylight_below_cpp(mock_hatch_data_high_latitude):
    """Test that hatching is zero if daylight is always below the critical photoperiod."""
    temp_data, lat_data = mock_hatch_data_high_latitude
    result = mosq_dia_hatch(temp_data, lat_data)
    # Expect all zeros because daylight hours will be below CPP for the first 60 days
    assert np.all(result.values == 0)


def test_mosq_dia_hatch_with_nan_input(mock_hatch_data):
    """Test that NaN values in the input are handled correctly."""
    temp_data, lat_data = mock_hatch_data
    C = CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING
    temp_data.values[:] = C["CTT"] + 10
    temp_data.values[0, 0, 30] = np.nan  # Introduce a NaN

    result = mosq_dia_hatch(temp_data, lat_data)
    # Ensure no NaNs are in the output
    assert not np.isnan(result.values).any()
    # The cell with the original NaN should result in 0 after nan_to_num
    assert result.values[0, 0, 30] == 0


def test_mosq_dia_hatch_with_dummy_data(temp_dummy_data):
    """
    Test mosq_dia_hatch with dummy data and compare against a known Octave result.
    """
    # Extract latitude coordinate from the input data
    latitude = temp_dummy_data.latitude

    # Run the function
    result = mosq_dia_hatch(temp_dummy_data, latitude)

    # Define the expected result from the Octave output
    # Note: The dimensions are transposed from (lat, lon, time) in Octave
    # to (lon, lat, time) as returned by the Python function.
    expected_octave_result = np.array(
        [
            # Time slice 1
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1, 0.1],
            ],
            # Time slice 2
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            # Time slice 3
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.1],
            ],
        ]
    )

    # Assert that the function's output matches the expected result
    np.testing.assert_allclose(result.values, expected_octave_result, atol=1e-4)


# ---- mosq_dia_lay()
def test_mosq_dia_lay_invalid_temp_dims_raises(mock_lay_data):
    """Test that a non-3D temperature array raises a ValueError."""
    _, lat_data = mock_lay_data
    invalid_temp = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Temperature array must be 3D"):
        mosq_dia_lay(invalid_temp, lat_data)


def test_mosq_dia_lay_invalid_lat_dims_raises(mock_lay_data):
    """Test that a non-1D latitude array raises a ValueError."""
    temp_data, _ = mock_lay_data
    invalid_lat = xr.DataArray(np.zeros((2, 2)), dims=("x", "y"))
    with pytest.raises(ValueError, match="Latitude array must be 1D."):
        mosq_dia_lay(temp_data, invalid_lat)


def test_mosq_dia_lay_mid_latitude_seasonal_split(mock_lay_data):
    """Test that diapause is zero in the first half of the year."""
    temp_data, lat_data = mock_lay_data
    result = mosq_dia_lay(temp_data, lat_data)

    # Assert first half of the year is all zero
    assert np.all(result.values[:, :, :HALF_DAYS_YEAR] == 0)
    # Assert there is some diapause in the second half
    assert np.any(result.values[:, :, HALF_DAYS_YEAR:] > 0)


def test_mosq_dia_lay_equator_no_diapause(mock_lay_data_equator):
    """Test that no diapause occurs at the equator."""
    temp_data, lat_data = mock_lay_data_equator
    result = mosq_dia_lay(temp_data, lat_data)
    # At the equator, daylight is always long enough to prevent diapause
    assert np.all(result.values == 0)


def test_mosq_dia_lay_polar_handling(mock_lay_data_polar):
    """Test that the function handles polar regions without producing NaNs."""
    temp_data, lat_data = mock_lay_data_polar
    result = mosq_dia_lay(temp_data, lat_data)
    # Ensure no NaNs are in the output, which can happen with extreme daylight calculations
    assert not np.isnan(result.values).any()


def test_mosq_dia_lay_output_values(mock_lay_data):
    """Test that all non-zero output values are equal to RATIO_DIA_LAY."""
    temp_data, lat_data = mock_lay_data
    result = mosq_dia_lay(temp_data, lat_data)
    ratio = CONSTANTS_MOSQUITO_DIAPAUSE_LAY["RATIO_DIA_LAY"]

    # Get all non-zero values from the result
    non_zero_values = result.values[result.values > 0]

    # Assert that all these non-zero values are equal to the expected ratio
    assert np.all(non_zero_values == ratio)
