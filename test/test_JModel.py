from model_backend import JModel
from pathlib import Path
import pandas as pd
import pytest


def test_jmodel_initialization():
    # Test valid initialization

    path = Path.cwd() / "test" / "test_r0.csv"

    model = JModel(
        input="input_data.csv",
        output="output_data.csv",
        r0_path=path,
        run_mode="allowed",
        grid_data_baseurl="https://example.com/grid_data",
        nuts_level=3,
        resolution="10M",
        year=2024,
    )

    assert model.input == "input_data.csv"
    assert model.output == "output_data.csv"
    assert model.run_mode == "allowed"
    assert isinstance(model.r0_data, pd.DataFrame)
    assert model.grid_data_baseurl == "https://example.com/grid_data"
    assert model.nuts_level == 3
    assert model.resolution == "10M"
    assert model.year == 2024
    assert model.min_temp == 8.6
    assert model.max_temp == 13.0

    with pytest.raises(ValueError):
        JModel(
            input=None,
            output="output_data.csv",
            r0_path=path,
            run_mode="invalid_forbiddenmode",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )
    with pytest.raises(ValueError):
        JModel(
            input="input_data.csv",
            output=None,
            r0_path=path,
            run_mode="forbidden",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )
    with pytest.raises(ValueError):
        JModel(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=None,
            run_mode="forbidden",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

    with pytest.raises(ValueError):
        JModel(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=path,
            run_mode="invalid_mode",  # Invalid run mode
            grid_data_baseurl="https://example.com/grid_data",
            nuts_level=3,
            resolution="10M",
            year=2024,
        )

    with pytest.raises(ValueError):
        JModel(
            input="input_data.csv",
            output="output_data.csv",
            r0_path=path,
            run_mode="parallel",  # Invalid run mode
            grid_data_baseurl=None,
            nuts_level=3,
            resolution="10M",
            year=2024,
        )


def test_model_read_input_data():
    pass


def test_model_run():
    pass
