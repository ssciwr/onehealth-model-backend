from heiplanet_models.utils import (
    read_geodata,
    detect_csr,
    load_module,
    load_name_from_module,
)
import geopandas as gpd
import pytest
from pathlib import Path


def test_read_geodata():
    # Test reading a sample geodata file
    data = read_geodata(
        nuts_level=3,
        year=2024,
        resolution="10M",
        base_url="https://gisco-services.ec.europa.eu/distribution/v2/nuts",
        url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
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
            url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}_INVALID.geojson",
        )


def test_detect_csr(make_test_data):
    # reassign coords
    with make_test_data as data:
        data = detect_csr(data)
        assert data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"


def test_detect_csr_invalid(make_invalid_test_data):
    with make_invalid_test_data as invalid_data:
        with pytest.raises(ValueError):
            detect_csr(invalid_data)


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
