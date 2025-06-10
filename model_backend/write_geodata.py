my_dict = {
    "band": "valid_time",
    "x": "longitude",
    "y": "latitude",
}


def write_to_geotiff(xarray_dataset, path_file: str, dict_variables, crs="EPSG:4326"):

    ds = xarray_dataset
    # validate it is 2D or 3D data

    # Write crs
    ds.rio.write_crs(crs, inplace=True)

    # Set spatial dimensions (if not automatically recognized, though rioxarray is good at this)
    ds = ds.rio.set_spatial_dims(
        x_dim=dict_variables["x"], y_dim=dict_variables["y"], inplace=True
    )

    ds = ds.rename({dict_variables["band"]: "band"})

    ds.rio.to_raster(path_file)
