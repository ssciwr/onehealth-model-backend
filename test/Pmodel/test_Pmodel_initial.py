"""Pytest unit tests for Pmodel_initial.py module functions.

Tests use dummy NetCDF datasets located in test/resources:
- dense_dummy.nc: human population density dataset
- pr_dummy.nc: rainfall dataset
- temperature_dummy.nc: temperature dataset
"""

import logging
import pytest
import numpy as np
import xarray as xr
from pathlib import Path

from src.heiplanet_models.Pmodel import Pmodel_initial

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Paths to dummy datasets
RESOURCES = Path(__file__).parent.parent / "resources"
DENSE_PATH = RESOURCES / "dense_dummy.nc"
RAIN_PATH = RESOURCES / "pr_dummy.nc"
TEMP_PATH = RESOURCES / "temperature_dummy.nc"


# ---- Pytest Fixtures
@pytest.fixture
def dummy_population():
    return Pmodel_initial.load_dataset(
        path_dataset=DENSE_PATH,
        variable_name="dens",
        names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
    )


@pytest.fixture
def dummy_rainfall():
    return Pmodel_initial.load_dataset(path_dataset=RAIN_PATH, variable_name="tp")


@pytest.fixture
def dummy_temperature():
    return Pmodel_initial.load_dataset(path_dataset=TEMP_PATH, variable_name="t2m")


@pytest.fixture
def misaligned_population_nc(tmp_path):
    """Creates a temporary population .nc file with custom longitude/latitude."""
    lon = np.linspace(0, 10, 5)
    lat = np.linspace(0, 20, 6)
    time = np.arange(2)
    data = np.random.rand(len(lon), len(lat), len(time))
    ds = xr.DataArray(
        data,
        coords={"longitude": lon, "latitude": lat, "time": time},
        dims=("longitude", "latitude", "time"),
        name="dens",
    ).to_dataset()
    nc_path = tmp_path / "misaligned_population.nc"
    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def misaligned_rainfall_nc(tmp_path):
    """Creates a temporary rainfall .nc file with different longitude/latitude."""
    lon = np.linspace(1, 11, 5)  # shifted by +1
    lat = np.linspace(2, 22, 6)  # shifted by +2
    time = np.arange(2)
    data = np.random.rand(len(lon), len(lat), len(time))
    ds = xr.DataArray(
        data,
        coords={"longitude": lon, "latitude": lat, "time": time},
        dims=("longitude", "latitude", "time"),
        name="tp",
    ).to_dataset()
    nc_path = tmp_path / "misaligned_rainfall.nc"
    ds.to_netcdf(nc_path)
    return nc_path


# ---- load_dataset()
def test_load_dataset_success():
    ds = Pmodel_initial.load_dataset(path_dataset=RAIN_PATH, variable_name="tp")
    assert isinstance(ds, xr.DataArray)
    assert ds.name == "tp"
    assert ds.size > 0


def test_load_dataset_missing_variable():
    with pytest.raises(KeyError):
        Pmodel_initial.load_dataset(path_dataset=RAIN_PATH, variable_name="not_a_var")


# ---- load_initial_conditions()
def test_load_initial_conditions_default():
    arr = Pmodel_initial.load_initial_conditions(filepath_previous=None, sizes=(5, 5))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 5, len(Pmodel_initial.MODEL_VARIABLES))
    # Check that only the second variable is initialized
    assert np.all(arr[:, :, 1] == Pmodel_initial.K1 * Pmodel_initial.K2)


# ---- load_temperature()
def test_load_temperature(dummy_temperature):
    temp, temp_mean = Pmodel_initial.load_temperature(
        path_dataset=TEMP_PATH, variable_name="t2m", time_step=2
    )
    assert isinstance(temp, np.ndarray)
    assert isinstance(temp_mean, xr.DataArray)
    # Check time axis was expanded
    assert temp.shape[2] == temp_mean.shape[2] * 2


# ---- align_xarray_datasets()
def test_misaligned_coordinates(misaligned_population_nc, misaligned_rainfall_nc):
    pop = Pmodel_initial.load_dataset(
        path_dataset=misaligned_population_nc, variable_name="dens"
    )
    rain = Pmodel_initial.load_dataset(
        path_dataset=misaligned_rainfall_nc, variable_name="tp"
    )
    # Assert longitude and latitude are misaligned
    assert not np.array_equal(pop.longitude.values, rain.longitude.values)
    assert not np.array_equal(pop.latitude.values, rain.latitude.values)


def test_align_population_to_rainfall(misaligned_population_nc, misaligned_rainfall_nc):
    # load misaligned datasets
    pop = Pmodel_initial.load_dataset(
        path_dataset=misaligned_population_nc,
        variable_name="dens",
    )
    rain = Pmodel_initial.load_dataset(
        path_dataset=misaligned_rainfall_nc,
        variable_name="tp",
    )

    # verify they have different longitudes and latitudes

    # Produce an aligned version of population based on rainfall coordinates
    pop_aligned = Pmodel_initial.align_xarray_datasets(pop, rain)
    logger.debug(f"Lon. rainfall: \t\t\t{rain.longitude.values}")
    logger.debug(f"Lon. population: \t\t{pop.longitude.values}")
    logger.debug(f"Lon. aligned population: \t{pop_aligned.longitude.values}")

    # Verify longitude and latitudes are different between datasets
    assert not np.allclose(pop.longitude.values, rain.longitude.values)
    assert not np.allclose(pop.latitude.values, rain.latitude.values)

    # Verify the dataset is aligned with respect rainfall dataset
    assert np.allclose(pop_aligned.longitude.values, rain.longitude.values)


def test_load_data():
    result = Pmodel_initial.load_data(
        path_temperature=TEMP_PATH,
        path_rainfall=RAIN_PATH,
        path_population=DENSE_PATH,
        time_step=1,
    )
    assert hasattr(result, "initial_conditions")
    assert hasattr(result, "latitude")
    assert hasattr(result, "population_density")
    assert hasattr(result, "rainfall")
    assert hasattr(result, "temperature")
    assert hasattr(result, "temperature_mean")
