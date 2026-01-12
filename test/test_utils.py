from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import xarray as xr


from heiplanet_models.utils import (
    read_geodata,
    detect_csr,
    load_module,
    load_name_from_module,
    validate_spatial_alignment,
)


# ---- Fixtures
def make_dataarray(lat, lon, name="var"):
    data = np.zeros((len(lat), len(lon)))
    return xr.DataArray(
        data,
        coords={"latitude": lat, "longitude": lon},
        dims=["latitude", "longitude"],
        name=name,
    )


# ---- read_geodata()
def test_read_geodata():
    # Test reading a sample geodata file
    data = read_geodata(
        nuts_level=3,
        year=2024,
        resolution="10M",
        base_url="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
        url=lambda base_url,
        resolution,
        year,
        nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
    )

    assert isinstance(data, gpd.GeoDataFrame), "Data should be a GeoDataFrame"
    assert not data.empty, "Data should not be empty"
    assert "geometry" in data.columns, "Data should contain a geometry column"

    with pytest.raises(
        RuntimeError,
    ):
        read_geodata(
            nuts_level=3,
            year=2024,
            resolution="10M",
            base_url="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
            url=lambda base_url,
            resolution,
            year,
            nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}_INVALID.geojson",
        )


# ---- detect_csr()
def test_detect_csr(make_test_data):
    # reassign coords
    with make_test_data as data:
        data = detect_csr(data)
        assert data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"


def test_detect_csr_invalid(make_invalid_test_data):
    with make_invalid_test_data as invalid_data:
        with pytest.raises(ValueError):
            detect_csr(invalid_data)


# ---- load_module()
def test_load_module():
    module_name = "cm"
    file_path = Path("./test/computation_module.py").resolve().absolute()
    module = load_module(module_name, file_path)
    assert module is not None, "Module should be loaded successfully"
    assert hasattr(module, "add"), "Module should have an 'add' function"

    with pytest.raises(RuntimeError, match="Error in loading module"):
        load_module("non_existent_module", "./non_existent_path.py")


def test_load_name_from_module():
    file_path = Path("./test/computation_module.py").resolve().absolute()
    function_name = "add"
    module_name = "cm"
    function = load_name_from_module(module_name, file_path, function_name)
    assert callable(function), "Loaded function should be callable"
    assert function(2, 3) == 5, "Function should return correct result"


# ---- validate_spatial_alignment
def test_validate_spatial_alignment_success():
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(-180, 180, 5)
    arr1 = make_dataarray(lat, lon)
    arr2 = make_dataarray(lat, lon)
    # Should not raise
    validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_latitude_mismatch():
    lat1 = np.linspace(-90, 90, 5)
    lat2 = np.linspace(-80, 80, 5)
    lon = np.linspace(-180, 180, 5)
    arr1 = make_dataarray(lat1, lon)
    arr2 = make_dataarray(lat2, lon)
    with pytest.raises(ValueError, match="must be aligned"):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_longitude_mismatch():
    lat = np.linspace(-90, 90, 5)
    lon1 = np.linspace(-180, 180, 5)
    lon2 = np.linspace(-170, 170, 5)
    arr1 = make_dataarray(lat, lon1)
    arr2 = make_dataarray(lat, lon2)
    with pytest.raises(ValueError, match="must be aligned"):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_missing_latitude():
    lon = np.linspace(-180, 180, 5)
    arr1 = xr.DataArray(np.zeros((5,)), coords={"longitude": lon}, dims=["longitude"])
    arr2 = make_dataarray(np.linspace(-90, 90, 5), lon)
    with pytest.raises(
        ValueError, match="Input DataArrays must have a 'latitude' coordinate."
    ):
        validate_spatial_alignment(arr1, arr2)


def test_validate_spatial_alignment_missing_longitude():
    lat = np.linspace(-90, 90, 5)
    arr1 = xr.DataArray(np.zeros((5,)), coords={"latitude": lat}, dims=["latitude"])
    arr2 = make_dataarray(lat, np.linspace(-180, 180, 5))
    with pytest.raises(
        ValueError, match="Input DataArrays must have a 'longitude' coordinate."
    ):
        validate_spatial_alignment(arr1, arr2)
