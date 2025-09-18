import pytest
import numpy as np
import xarray as xr

from heiplanet_models.Pmodel.Pmodel_input import PmodelInput


# ===================================
# ===       Pytest Fixtures       ===
# ===================================
@pytest.fixture
def dummy_pmodel_input():
    """Fixture for a fully populated PmodelInput instance."""
    initial_conditions = np.zeros((2, 2, 3))
    latitude = xr.DataArray(np.linspace(-10, 10, 2), dims="latitude")
    population_density = xr.DataArray(np.ones((2, 2)), dims=("longitude", "latitude"))
    rainfall = xr.DataArray(np.zeros((2, 2)), dims=("longitude", "latitude"))
    temperature = xr.DataArray(
        np.full((2, 2, 3), 25.0), dims=("longitude", "latitude", "time")
    )
    temperature_mean = xr.DataArray(
        np.full((2, 2, 3), 20.0), dims=("longitude", "latitude", "time")
    )
    return PmodelInput(
        initial_conditions=initial_conditions,
        latitude=latitude,
        population_density=population_density,
        rainfall=rainfall,
        temperature=temperature,
        temperature_mean=temperature_mean,
    )


# ===================================
# ===         Unit tests          ===
# ===================================
# ---- Unit Tests for PmodelInput
def test_pmodelinput_instantiation_and_types(dummy_pmodel_input):
    assert isinstance(dummy_pmodel_input, PmodelInput)
    assert isinstance(dummy_pmodel_input.initial_conditions, np.ndarray)
    assert isinstance(dummy_pmodel_input.latitude, xr.DataArray)
    assert isinstance(dummy_pmodel_input.population_density, xr.DataArray)
    assert isinstance(dummy_pmodel_input.rainfall, xr.DataArray)
    assert isinstance(dummy_pmodel_input.temperature, xr.DataArray)
    assert isinstance(dummy_pmodel_input.temperature_mean, xr.DataArray)


# Test __repr__ output
def test_pmodelinput_repr_contains_class_and_attrs(dummy_pmodel_input):
    rep = repr(dummy_pmodel_input)
    assert "PmodelInput" in rep
    for attr in [
        "initial_conditions",
        "latitude",
        "population_density",
        "rainfall",
        "temperature",
        "temperature_mean",
    ]:
        assert attr in rep


# Test missing required attributes raises TypeError
def test_pmodelinput_missing_required_attributes():
    import pytest

    with pytest.raises(TypeError):
        PmodelInput()
