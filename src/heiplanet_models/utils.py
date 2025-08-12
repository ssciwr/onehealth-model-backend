from __future__ import annotations

import geopandas as gpd
import xarray as xr
import importlib


def detect_csr(data: xr.Dataset) -> xr.Dataset:
    """Detects and sets the coordinate reference system (CRS) for an xarray dataset. Uses rioxarray to handle the CRS. If the crs is not defined, it checks if the coordinates match the expected ranges for EPSG:4326 (standard lat/lon coordinates).

    Args:
        data (xr.Dataset): xarray dataset to check and set the CRS for. typically these are era5 data or other climate data which often do not come with a given crs. Currently, this only supports the
        EPSG.4326 standard lat/lon coordinates, which are defined as follows:
        - Longitude: -180 to 180 degrees
        - Latitude: -90 to 90 degrees
        The spatial coordinates of the dataset must be called 'latitude' and 'longitude'.

    Raises:
        ValueError: When the CRS is not defined and the coordinates do not match the expected ranges for EPSG:4326.

    Returns:
        xr.Dataset: dataset with the CRS set to EPSG:4326 if it was not already defined and the coordinates match the expected ranges.

    """

    # this currently only detects EPSG:4326 standard lat/lon coordinates
    if (
        -181.0 < data.longitude.min().values < -179.0
        and 179.0 < data.longitude.max().values < 181.0
        and -91.0 < data.latitude.min().values < -89.0
        and 89.0 < data.latitude.max().values < 91.0
    ):
        data = data.rio.write_crs("EPSG:4326")
    else:
        raise ValueError(
            "Coordinate reference system (CRS) is not defined and coordinates do not match expected ranges for EPSG:4326."
        )
    return data


def read_geodata(
    nuts_level: int = 3,
    year: int = 2024,
    resolution: str = "10M",
    base_url: str = "https://gisco-services.ec.europa.eu/distribution/v2/nuts",
    url: callable = lambda base_url, resolution, year, nuts_level: f"{base_url}/geojson/NUTS_RG_{resolution}_{year}_4326_LEVL_{nuts_level}.geojson",
):
    """load Eurostat NUTS geospatial data from the Eurostat service.

    Args:
        nuts_level (int, optional): nuts administrative region level. Defaults to 3.
        year (int, optional): year to load data for. Defaults to 2024.
        resolution (str, optional): resolution of the map. Resolution of the geospatial data. One of
        "60" (1:60million),
        "20" (1:20million)
        "10" (1:10million)
        "03" (1:3million) or
        "01" (1:1million).
        Defaults to "10M".
        base_url (str, optional): _description_. Defaults to "https://gisco-services.ec.europa.eu/distribution/v2/nuts".
        url (callable, optional): builds the full url from the arguments passed to the function.must have the signature url(base_url, resolution, year, nuts_level).

    Returns:
        geopandas.dataframe: Dataframe containing the NUTS geospatial data.
    """
    url_str = url(
        nuts_level=nuts_level, year=year, resolution=resolution, base_url=base_url
    )

    try:
        nuts_data = gpd.read_file(url_str)
        return nuts_data
    except Exception as e:
        raise RuntimeError(f"Failed to download from {url_str}: {e}")


# helpers for loading code
def load_module(module_name: str, file_path: str):
    """
    load_module Load a python module from 'path' with alias 'alias'

    Args:
        module_name (str): module alias.
        file_path (str): Path to load the module from

    Returns:
        module: Python module that has been loaded
    """
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error in loading module {file_path}") from e
    return module


def load_name_from_module(module_name: str, file_path: str, name: str):
    """
    load_name_from_module Load a python module from 'path' with alias 'alias'

    Args:
        module_name (str): module alias.
        file_path (str): Path to load the module from
        name (str): name to import
    Returns:
        module: Python module that has been loaded
    """
    module = load_module(module_name, file_path)
    return getattr(module, name)
