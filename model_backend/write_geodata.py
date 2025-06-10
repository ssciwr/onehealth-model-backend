"""
Module for writing xarray datasets to GeoTIFF files using rioxarray.
"""

from typing import Dict
import xarray as xr
import logging


def write_to_geotiff(
    xarray_dataset: xr.DataArray,
    path_file: str,
    dict_variables: Dict[str, str],
    crs: str = "EPSG:4326",
) -> None:
    """
    Write an xarray DataArray or Dataset to a GeoTIFF file.

    Args:
        xarray_dataset (xr.DataArray): The data to write.
        path_file (str): Output file path.
        dict_variables (dict): Mapping for band, x, and y dimension names.
        crs (str, optional): Coordinate Reference System. Defaults to "EPSG:4326".

    Raises:
        ValueError: If input data is not 2D or 3D.
        Exception: For any other errors during writing.
    """
    ds = xarray_dataset
    try:
        # Validate dimensions
        if ds.ndim not in (2, 3):
            raise ValueError("Input data must be 2D or 3D for GeoTIFF export.")

        # Write CRS
        ds = ds.rio.write_crs(crs, inplace=False)

        # Set spatial dimensions
        ds = ds.rio.set_spatial_dims(
            x_dim=dict_variables["x"], y_dim=dict_variables["y"], inplace=False
        )

        # Rename band dimension if needed (only if present in both dict and dims)
        band_key = dict_variables.get("band")
        if band_key and band_key in ds.dims:
            ds = ds.rename({band_key: "band"})

        ds.rio.to_raster(path_file)
        logging.info(f"GeoTIFF written to {path_file}")

    except Exception as e:
        logging.error(f"Failed to write GeoTIFF: {e}")
        raise
