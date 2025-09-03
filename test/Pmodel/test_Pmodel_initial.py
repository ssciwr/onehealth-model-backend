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
def dummy_dataset_path():
    """Provides a path to a dummy dataset for testing."""
    return DENSE_PATH


@pytest.fixture
def dummy_population():
    return Pmodel_initial.load_dataset(
        path_dataset=DENSE_PATH,
        variable_name="dens",
        # names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
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


@pytest.fixture
def initial_conditions_file(tmp_path, misaligned_population_nc):
    """Creates a temporary NetCDF file simulating previous initial conditions."""
    # Load the dummy population data
    dens = xr.open_dataset(misaligned_population_nc)["dens"]
    # Create a dataset with all required model variables
    ds = xr.Dataset({var: dens for var in Pmodel_initial.MODEL_VARIABLES})
    nc_path = tmp_path / "initial_conditions.nc"
    ds.to_netcdf(nc_path)
    return nc_path


# ---- load_dataset()
def test_load_dataset_success(misaligned_rainfall_nc):
    ds = Pmodel_initial.load_dataset(
        path_dataset=misaligned_rainfall_nc, variable_name="tp"
    )
    assert isinstance(ds, xr.DataArray)
    assert ds.name == "tp"
    assert ds.size > 0


def test_load_dataset_missing_variable(misaligned_rainfall_nc):
    with pytest.raises(KeyError):
        Pmodel_initial.load_dataset(
            path_dataset=misaligned_rainfall_nc, variable_name="not_a_var"
        )


# ---- load_initial_conditions()
def test_load_initial_conditions_default():
    arr = Pmodel_initial.load_initial_conditions(filepath_previous=None, sizes=(5, 5))
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (5, 5, len(Pmodel_initial.MODEL_VARIABLES))
    # Check that only the second variable is initialized
    assert np.all(arr[:, :, 1] == Pmodel_initial.CONST_K1 * Pmodel_initial.CONST_K2)


def test_load_initial_conditions_from_file(
    misaligned_population_nc, initial_conditions_file
):
    arr = Pmodel_initial.load_initial_conditions(
        filepath_previous=initial_conditions_file,
        sizes=(5, 6),  # Use shape matching your fixture
    )
    assert arr.shape == (5, 6, len(Pmodel_initial.MODEL_VARIABLES))


# ---- load_temperature()
def test_load_temperature(misaligned_population_nc):
    temp, temp_mean = Pmodel_initial.load_temperature(
        path_dataset=misaligned_population_nc, variable_name="dens", time_step=2
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
    pop = Pmodel_initial.load_dataset(
        path_dataset=misaligned_population_nc, variable_name="dens"
    )
    rain = Pmodel_initial.load_dataset(
        path_dataset=misaligned_rainfall_nc, variable_name="tp"
    )
    pop_aligned = Pmodel_initial.align_xarray_datasets(pop, rain)
    # Verify longitude and latitudes are different between datasets
    assert not np.allclose(pop.longitude.values, rain.longitude.values)
    assert not np.allclose(pop.latitude.values, rain.latitude.values)

    # Verify the dataset is aligned with respect rainfall dataset
    assert np.allclose(pop_aligned.longitude.values, rain.longitude.values)
    assert np.allclose(pop_aligned.latitude.values, rain.latitude.values)


def test_load_data(misaligned_population_nc, misaligned_rainfall_nc):
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


def test_load_dataset_invalid_path():
    with pytest.raises(Exception):
        Pmodel_initial.load_dataset(path_dataset="nonexistent.nc")


def test_load_initial_conditions_invalid_file(tmp_path):
    invalid_path = tmp_path / "invalid.nc"
    invalid_path.write_text("not a netcdf")
    with pytest.raises(Exception):
        Pmodel_initial.load_initial_conditions(
            filepath_previous=invalid_path, sizes=(2, 2)
        )


def test_load_dataset_no_rename():
    ds = Pmodel_initial.load_dataset(path_dataset=DENSE_PATH, variable_name="dens")
    assert isinstance(ds, xr.DataArray)


def test_load_dataset_with_rename():
    ds = Pmodel_initial.load_dataset(
        path_dataset=DENSE_PATH,
        names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
        variable_name="dens",
    )
    # Should still return a DataArray, but with renamed dims
    assert isinstance(ds, xr.DataArray)


def test_load_dataset_with_transpose():
    ds = Pmodel_initial.load_dataset(
        path_dataset=DENSE_PATH,
        names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
        dimension_order=("longitude", "latitude", "time"),
        variable_name="dens",
    )
    assert isinstance(ds, xr.DataArray)


def test_load_dataset_variable_not_found():
    with pytest.raises(KeyError):
        Pmodel_initial.load_dataset(path_dataset=DENSE_PATH, variable_name="not_a_var")


def test_load_initial_conditions_file_not_exists(tmp_path):
    fake_path = tmp_path / "not_exists.nc"
    arr = Pmodel_initial.load_initial_conditions(
        filepath_previous=fake_path, sizes=(2, 2)
    )
    assert arr.shape == (2, 2, len(Pmodel_initial.MODEL_VARIABLES))


def test_load_initial_conditions_missing_var():
    # Create a dataset missing one variable
    ds = xr.open_dataset(DENSE_PATH)
    ds = ds.drop_vars("dens")
    missing_var_path = DENSE_PATH.parent / "missing.nc"
    ds.to_netcdf(missing_var_path)
    with pytest.raises(KeyError):
        Pmodel_initial.load_initial_conditions(
            filepath_previous=missing_var_path, sizes=(5, 6)
        )


def test_load_temperature_invalid():
    # Should work for valid input
    temp, temp_mean = Pmodel_initial.load_temperature(
        path_dataset=TEMP_PATH, variable_name="t2m", time_step=2
    )
    assert temp.shape[2] == temp_mean.shape[2] * 2


def test_align_xarray_datasets_success():
    ds = Pmodel_initial.load_dataset(
        DENSE_PATH,
        names_dimensions={"lon": "longitude", "lat": "latitude", "time": "time"},
    )
    arr1 = ds["dens"]
    arr2 = ds["dens"]
    aligned = Pmodel_initial.align_xarray_datasets(arr1, arr2)
    assert isinstance(aligned, xr.DataArray)


def test_align_xarray_datasets_fail():
    ds = xr.open_dataset(DENSE_PATH)
    arr1 = ds["dens"]
    # Remove latitude from arr2 to force failure
    arr2 = arr1.drop_vars("lat")
    with pytest.raises(Exception):
        Pmodel_initial.align_xarray_datasets(arr1, arr2)


def test_load_data_success():
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


def test_load_data_fail(tmp_path):
    # Use a non-existent file to trigger error
    fake_path = tmp_path / "not_exists.nc"
    with pytest.raises(Exception):
        Pmodel_initial.load_data(
            path_temperature=fake_path,
            path_rainfall=fake_path,
            path_population=fake_path,
            time_step=1,
        )


def test_main_runs(tmp_path, caplog):
    # Create a dummy file for rainfall
    lon = np.linspace(0, 10, 2)
    lat = np.linspace(0, 20, 2)
    time = np.arange(2)
    data = np.random.rand(len(lon), len(lat), len(time))
    ds = xr.DataArray(
        data,
        coords={"longitude": lon, "latitude": lat, "time": time},
        dims=("longitude", "latitude", "time"),
        name="tp",
    ).to_dataset()
    nc_path = tmp_path / "rainfall.nc"
    ds.to_netcdf(nc_path)
    # Patch config to use this file
    Pmodel_initial.PATH_DATASETS_SANDBOX["RAINFALL"] = nc_path
    # Run main (should not raise)
    Pmodel_initial.main()


def test_load_dataset_rename_fail(dummy_dataset_path, caplog):
    """Test that load_dataset raises an error if renaming fails."""
    with pytest.raises(Exception):
        Pmodel_initial.load_dataset(
            path_dataset=dummy_dataset_path,
            names_dimensions={"non_existent_dim": "new_dim"},
        )
    assert "Failed to rename dimensions" in caplog.text


def test_load_dataset_transpose_fail(dummy_dataset_path, caplog):
    """Test that load_dataset raises an error if transposing fails."""
    with pytest.raises(Exception):
        Pmodel_initial.load_dataset(
            path_dataset=dummy_dataset_path,
            dimension_order=("longitude", "latitude", "non_existent_dim"),
        )
    assert "Failed to transpose dataset" in caplog.text


def test_load_initial_conditions_extract_fail(
    initial_conditions_file, monkeypatch, caplog
):
    """Test that load_initial_conditions raises an error if extracting a variable fails."""

    # Mock isel to raise an exception
    def mock_isel(*args, **kwargs):
        raise IndexError("Mock isel failure")

    monkeypatch.setattr(xr.DataArray, "isel", mock_isel)

    with pytest.raises(Exception):
        Pmodel_initial.load_initial_conditions(
            filepath_previous=initial_conditions_file, sizes=(10, 10)
        )
    assert "Failed to extract variable" in caplog.text


def test_load_temperature_expand_fail(monkeypatch, caplog):
    """Test that load_temperature raises an error if np.repeat fails."""

    # Mock np.repeat to raise an exception
    def mock_repeat(*args, **kwargs):
        raise ValueError("Mock repeat failure")

    monkeypatch.setattr(np, "repeat", mock_repeat)

    with pytest.raises(Exception):
        Pmodel_initial.load_temperature(path_dataset=TEMP_PATH, variable_name="t2m")
    assert "Failed to expand temperature array" in caplog.text


def test_main_execution_fail(monkeypatch, caplog):
    """Test that the main function logs an error on failure."""

    # Mock load_dataset to fail
    def mock_load_dataset(*args, **kwargs):
        raise ValueError("Mock loading failure")

    monkeypatch.setattr(
        "src.heiplanet_models.Pmodel.Pmodel_initial.load_dataset", mock_load_dataset
    )

    Pmodel_initial.main()
    assert "Error in main execution: Mock loading failure" in caplog.text


def test_load_dataset_return_full_dataset(dummy_dataset_path):
    """Test load_dataset returns a full xr.Dataset when no variable_name is given."""
    # This test covers the 'else' path for variable extraction.
    ds = Pmodel_initial.load_dataset(path_dataset=dummy_dataset_path)
    assert isinstance(ds, xr.Dataset)
    assert "dens" in ds.data_vars


def test_load_data_rainfall_load_fails(monkeypatch, caplog):
    """Test that load_data handles a failure specifically when loading rainfall."""
    # Mock load_dataset to fail only when loading the rainfall file.
    original_load = Pmodel_initial.load_dataset

    def mock_load_fail_on_rainfall(*args, **kwargs):
        if "pr_dummy" in str(kwargs.get("path_dataset", "")):
            raise ValueError("Mock rainfall loading failure")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(
        "src.heiplanet_models.Pmodel.Pmodel_initial.load_dataset",
        mock_load_fail_on_rainfall,
    )

    with pytest.raises(Exception):
        Pmodel_initial.load_data()
    # The outer exception handler logs a more general message.
    # We assert for this final message.
    assert "Failed to load model input data" in caplog.text


def test_align_xarray_datasets_alignment_error(
    dummy_rainfall, misaligned_population_nc, monkeypatch, caplog
):
    """Test that align_xarray_datasets handles an exception from interpolation."""

    # Mock the .interp() method on the DataArray class to raise an error
    def mock_interp(*args, **kwargs):
        raise RuntimeError("Mock internal interpolation error")

    monkeypatch.setattr(xr.DataArray, "interp", mock_interp)

    # Load the misaligned dataset into a DataArray before passing it
    misaligned_pop_data = Pmodel_initial.load_dataset(
        path_dataset=misaligned_population_nc, variable_name="dens"
    )

    with pytest.raises(Exception):
        Pmodel_initial.align_xarray_datasets(misaligned_pop_data, dummy_rainfall)
    assert "Failed to align coordinates using interpolation" in caplog.text
