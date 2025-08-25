import numpy as np
import pytest

from src.model_backend.Pmodel.functions import (
    revolution_angle,
    declination_angle,
    daylight_forsythe,
)


# ---- revolution_angle()
def test_revolution_angle_basic():
    # Test for a typical day value
    angle = revolution_angle(100)
    assert isinstance(angle, float)
    assert -np.pi <= angle <= np.pi


# ---- declination_angle()
def test_declination_angle_basic():
    # Test for a typical revolution angle
    rev_angle = revolution_angle(100)
    decl_angle = declination_angle(rev_angle)
    assert isinstance(decl_angle, float)
    assert -np.pi / 2 <= decl_angle <= np.pi / 2


# ---- daylight_forsythe()
def test_daylight_forsythe_basic():
    # Test for typical values
    latitude = 45.0
    rev_angle = revolution_angle(100)
    decl_angle = declination_angle(rev_angle)
    daylight = daylight_forsythe(latitude, decl_angle, 0)
    assert isinstance(daylight, float)
    assert 0 <= daylight <= 24


def test_daylight_forsythe_invalid_latitude():
    # Latitude out of bounds should raise ValueError
    rev_angle = revolution_angle(100)
    decl_angle = declination_angle(rev_angle)
    with pytest.raises(ValueError):
        daylight_forsythe(100.0, decl_angle, 0)


# ---- mosq_dia_lay()

# ---- mosq_dia_hatch()

# ---- mosq_surv_ed()
