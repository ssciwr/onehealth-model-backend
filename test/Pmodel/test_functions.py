import pytest
import numpy as np
import xarray as xr

from model_backend.Pmodel.Pmodel_functions import (
    carrying_capacity,
    water_hatching,
)


# -------------------------------
# ----    Pytest Fixtures    ----
# -------------------------------
@pytest.fixture
def dummy_constants_carrying_capacity():
    return {
        "ALPHA_RAIN": 2.0,
        "ALPHA_DENS": 3.0,
        "GAMMA": 0.5,
        "LAMBDA": 1.5,
    }


@pytest.fixture
def dummy_constants_water_hatching():
    return {
        "E_OPT": 8.0,
        "E_VAR": 0.05,
        "E_0": 1.5,
        "E_RAT": 0.2,
        "E_DENS": 0.01,
        "E_FAC": 0.01,
    }


@pytest.fixture
def rainfall_data():
    # Create a simple rainfall DataArray: shape (time, lat, lon)
    data = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
            [[9, 10], [11, 12]],
        ]
    )
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [0, 1, 2],
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def population_data_time_variant():
    # Population density with time dimension
    data = np.array(
        [
            [[10, 20], [30, 40]],
            [[50, 60], [70, 80]],
            [[90, 100], [110, 120]],
        ]
    )
    return xr.DataArray(
        data,
        dims=("time", "latitude", "longitude"),
        coords={
            "time": [0, 1, 2],
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


@pytest.fixture
def population_data_time_invariant():
    # Population density without time dimension
    data = np.array([[10, 20], [30, 40]])
    return xr.DataArray(
        data,
        dims=("latitude", "longitude"),
        coords={
            "latitude": [10, 20],
            "longitude": [30, 40],
        },
    )


# ----------------------------------------------
# ----              Unit Tests              ----
# ----------------------------------------------


# ---------------------------------------
# ----    Carrying Capacity Tests    ----
# ---------------------------------------
def test_capacity_time_variant_population(
    rainfall_data, population_data_time_variant, dummy_constants_carrying_capacity
):
    result = carrying_capacity(
        rainfall_data, population_data_time_variant, dummy_constants_carrying_capacity
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data.shape
    # Check that result is not all zeros
    assert np.any(result.values != 0)


def test_capacity_time_invariant_population(
    rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
):
    result = carrying_capacity(
        rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
    )
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data.shape
    # Check that result is not all zeros
    assert np.any(result.values != 0)


def test_capacity_raises_with_missing_constants(
    rainfall_data, population_data_time_invariant
):
    incomplete_constants = {"ALPHA_RAIN": 1.0}
    with pytest.raises(KeyError):
        # Should raise because required keys are missing
        carrying_capacity(
            rainfall_data, population_data_time_invariant, incomplete_constants
        )


def test_capacity_scaling_effect(
    rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
):
    # Test that changing LAMBDA scales the result
    result1 = carrying_capacity(
        rainfall_data,
        population_data_time_invariant,
        {**dummy_constants_carrying_capacity, "LAMBDA": 1.0},
    )
    result2 = carrying_capacity(
        rainfall_data,
        population_data_time_invariant,
        {**dummy_constants_carrying_capacity, "LAMBDA": 10.0},
    )
    np.testing.assert_allclose(result2.values, result1.values * 10)


def test_capacity_output_consistency(
    rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
):
    # The function should be deterministic
    result1 = carrying_capacity(
        rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
    )
    result2 = carrying_capacity(
        rainfall_data, population_data_time_invariant, dummy_constants_carrying_capacity
    )
    xr.testing.assert_allclose(result1, result2)


# ---------------------------------------
# ----    Water Hatching Tests       ----
# ---------------------------------------
def test_water_hatching_time_variant_population(
    rainfall_data, population_data_time_variant, dummy_constants_water_hatching
):
    # Convert keys to uppercase to match function expectations
    constants = {k: v for k, v in dummy_constants_water_hatching.items()}
    result = water_hatching(rainfall_data, population_data_time_variant, constants)
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data.shape
    # Should not be all zeros or NaNs
    assert np.any(result.values != 0)
    assert not np.any(np.isnan(result.values))


def test_water_hatching_time_invariant_population(
    rainfall_data, population_data_time_invariant, dummy_constants_water_hatching
):
    constants = {k: v for k, v in dummy_constants_water_hatching.items()}
    result = water_hatching(rainfall_data, population_data_time_invariant, constants)
    assert isinstance(result, xr.DataArray)
    assert result.shape == rainfall_data.shape
    assert np.any(result.values != 0)
    assert not np.any(np.isnan(result.values))


def test_water_hatching_raises_with_missing_constants(
    rainfall_data, population_data_time_invariant
):
    incomplete_constants = {"E_OPT": 1.0}
    with pytest.raises(KeyError):
        water_hatching(
            rainfall_data, population_data_time_invariant, incomplete_constants
        )


def test_water_hatching_weighted_combination(
    rainfall_data, population_data_time_invariant, dummy_constants_water_hatching
):
    # Test that changing E_RAT changes the weighting between pr_hatch and dens_adj
    constants_low = {k.upper(): v for k, v in dummy_constants_water_hatching.items()}
    constants_high = {**constants_low, "E_RAT": 0.9}
    constants_low["E_RAT"] = 0.1

    result_low = water_hatching(
        rainfall_data, population_data_time_invariant, constants_low
    )
    result_high = water_hatching(
        rainfall_data, population_data_time_invariant, constants_high
    )
    # The results should not be identical
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(result_low.values, result_high.values)


def test_water_hatching_output_consistency(
    rainfall_data, population_data_time_invariant, dummy_constants_water_hatching
):
    constants = {k: v for k, v in dummy_constants_water_hatching.items()}
    result1 = water_hatching(rainfall_data, population_data_time_invariant, constants)
    result2 = water_hatching(rainfall_data, population_data_time_invariant, constants)
    xr.testing.assert_allclose(result1, result2)
