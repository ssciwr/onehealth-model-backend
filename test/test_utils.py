from model_backend.utils import read_geodata, detect_csr
import geopandas as gpd
import pytest


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
        data = read_geodata(
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
