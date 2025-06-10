import geopandas as gpd
import xarray as xr
import numpy as np

from model_backend.read_geodata import read_netcdf4_format, read_nuts_data


# -----------------  Tests for read_netcdf4_format()
def test_read_netcdf4_format_file_not_exist():
    """Test that the function returns None when the file does not exist."""

    result = read_netcdf4_format("not_existing_file.nc")
    assert result is None


def test_read_netcdf4_format_valid_file(tmp_path):
    """Test that a valid NetCDF4 file is read and returned as an xarray.Dataset."""

    data = xr.Dataset({"row": (("x",), np.arange(3))})
    file_path = tmp_path / "test.nc"

    data.to_netcdf(file_path, engine="netcdf4")
    ds = read_netcdf4_format(str(file_path))
    assert isinstance(ds, xr.Dataset)
    assert "row" in ds
    assert np.all(ds["row"].values == np.arange(3))


def test_read_netcdf4_format_invalid_file(tmp_path):
    """Test that the function returns None when the file is not a valid NetCDF4 file."""

    file_path = tmp_path / "not_a_netcdf.txt"
    file_path.write_text("not a netcdf file")
    result = read_netcdf4_format(str(file_path))
    assert result is None


def test_read_netcdf4_format_empty_file(tmp_path):
    """Test that the function returns None when the file is empty."""
    file_path = tmp_path / "empty.nc"
    file_path.write_bytes(b"")
    result = read_netcdf4_format(str(file_path))
    assert result is None


# -----------------  Tests for read_nuts_data()
def test_read_nuts_data_file_not_exist():
    """Should return None if file does not exist."""
    result = read_nuts_data("not_existing_file.geojson")
    assert result is None


def test_read_nuts_data_empty_file(tmp_path):
    """Should return None if file is empty."""
    empty_file = tmp_path / "empty.geojson"
    empty_file.write_text("")
    result = read_nuts_data(str(empty_file))
    assert result is None


def test_read_nuts_data_invalid_file(tmp_path):
    """Should return None if file is not a valid geospatial file."""
    invalid_file = tmp_path / "invalid.geojson"
    invalid_file.write_text("not a geojson")
    result = read_nuts_data(str(invalid_file))
    assert result is None


def test_read_nuts_data_valid_geojson(tmp_path):
    """Should return a GeoDataFrame for a valid GeoJSON file."""
    geojson_content = """
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "geometry": {"type": "Point", "coordinates": [102.0, 0.5]},
          "properties": {"prop0": "value0"}
        }
      ]
    }
    """
    valid_file = tmp_path / "valid.geojson"
    valid_file.write_text(geojson_content)
    result = read_nuts_data(str(valid_file))
    assert isinstance(result, gpd.GeoDataFrame)
    assert not result.empty
