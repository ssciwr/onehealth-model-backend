from model_backend.utils import read_geodata
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
        RuntimeError, "Failed to download from url https://very_much_invalid/"
    ):
        data = read_geodata(
            nuts_level=3,
            year=2024,
            resolution="10M",
            base_url="https://very_much_invalid/",
            url=lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
        )

    # todo
