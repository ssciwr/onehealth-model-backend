import pytest
import math


from heiplanet_models.Pmodel.Pmodel_rates_birth import revolution_angle


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
