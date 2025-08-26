import pytest
import numpy as np
import xarray as xr

from src.heiplanet_models.Pmodel.Pmodel_input import PmodelInput


@pytest.fixture
def dummy_pmodel_input():
    """Fixture for a fully populated PmodelInput instance."""
    initial_conditions = np.zeros((2, 2, 3))
    latitude = xr.DataArray(np.linspace(-10, 10, 2), dims="longitude")
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


def test_pmodelinput_instantiation(dummy_pmodel_input):
    assert isinstance(dummy_pmodel_input, PmodelInput)
    assert isinstance(dummy_pmodel_input.initial_conditions, np.ndarray)
    assert isinstance(dummy_pmodel_input.latitude, xr.DataArray)
    assert isinstance(dummy_pmodel_input.population_density, xr.DataArray)
    assert isinstance(dummy_pmodel_input.rainfall, xr.DataArray)
    assert isinstance(dummy_pmodel_input.temperature, xr.DataArray)
    assert isinstance(dummy_pmodel_input.temperature_mean, xr.DataArray)


def test_print_attributes_output(dummy_pmodel_input, capsys):
    dummy_pmodel_input.print_attributes()
    captured = capsys.readouterr()
    # Check that all attributes are printed
    for attr in dummy_pmodel_input.__annotations__:
        assert attr in captured.out
    assert "PmodelInput Attributes:" in captured.out


def test_print_attributes_with_extra_attribute(dummy_pmodel_input, capsys):
    # Add an extra attribute not in __annotations__
    dummy_pmodel_input.extra_attr = "extra"
    dummy_pmodel_input.print_attributes()
    captured = capsys.readouterr()
    assert "extra_attr" in captured.out
    assert "str" in captured.out


def test_print_attributes_not_set_branch(capsys):
    # Create instance without temperature_mean to trigger "Not set" branch
    initial_conditions = np.zeros((2, 2, 3))
    latitude = xr.DataArray(np.linspace(-10, 10, 2), dims="longitude")
    population_density = xr.DataArray(np.ones((2, 2)), dims=("longitude", "latitude"))
    rainfall = xr.DataArray(np.zeros((2, 2)), dims=("longitude", "latitude"))
    temperature = xr.DataArray(
        np.full((2, 2, 3), 25.0), dims=("longitude", "latitude", "time")
    )
    # Do not set temperature_mean
    obj = PmodelInput(
        initial_conditions=initial_conditions,
        latitude=latitude,
        population_density=population_density,
        rainfall=rainfall,
        temperature=temperature,
        temperature_mean=None,  # Explicitly set to None
    )
    # Remove attribute to simulate missing
    del obj.temperature_mean
    obj.print_attributes()
    captured = capsys.readouterr()
    assert "temperature_mean: Not set" in captured.out
