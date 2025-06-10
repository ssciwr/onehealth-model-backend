from model_backend.utils import read_geodata, detect_csr
import geopandas as gpd
import pytest
import xarray as xr
import numpy as np


def make_test_data(defect=False) -> xr.Dataset:
    if defect:
        lon = np.linspace(-120, 120, 10)
        lat = np.linspace(-70, 76, 10)
    else:
        lon = np.linspace(-180, 180, 10)
        lat = np.linspace(-90, 90, 10)

    # Create the dataset with integer indices as coordinates
    data = np.random.rand(10, 10)  # Use y,x order for data array

    ds = xr.Dataset(
        data_vars={
            "temp": (("y", "x"), data),  # Use y,x order for dims
        },
        coords={
            "y": lon,  # Integer indices
            "x": lat,  # Integer indices
        },
    )

    return ds


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


def test_detect_csr(tmp_path):
    data = make_test_data()
    # write and read to convert to rioxarray raster
    data.to_netcdf(tmp_path / "test_data.nc")
    data = xr.open_dataset(tmp_path / "test_data.nc", engine="rasterio")

    # reassign coords
    lon = np.linspace(-180, 180, 10)
    lat = np.linspace(-90, 90, 10)
    data = data.assign_coords({"x": lon, "y": lat})
    data = detect_csr(data)

    assert data.rio.crs == "EPSG:4326", "CRS should be set to EPSG:4326"

    invalid_data = make_test_data(defect=True)

    with pytest.raises(ValueError):
        detect_csr(invalid_data)
