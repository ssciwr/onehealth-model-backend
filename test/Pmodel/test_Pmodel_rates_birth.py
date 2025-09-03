import pytest
import math

import numpy as np


from heiplanet_models.Pmodel.Pmodel_rates_birth import (
    revolution_angle,
    declination_angle,
    daylight_forsythe,
)
from heiplanet_models.Pmodel.Pmodel_params import CONSTANT_DECLINATION_ANGLE


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


# ---- declination_angle()
def test_declination_angle_at_equinox():
    """Test declination at an equinox (revolution angle PI/2), should be 0."""
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
