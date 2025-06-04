import os

import pandas as pd
from sandbox.edwin.interpolators import interpolator_function

FILENAME_R0_POSTERIOR_STATS = "R0_pip_stats.csv"
PATH_R0_POSTERIOR_STATS = os.path.join("./", FILENAME_R0_POSTERIOR_STATS)

X_COLUMN_STATS = "Temperature"
Y_COLUMN_STATS = "Median_R0"


# Reads the dataframe with posterior statistics
def _read_R0_posterior_stats(filepath):
    return pd.read_csv(filepath)


def create_interpolator():
    # Read the posterior stats
    df = _read_R0_posterior_stats(PATH_R0_POSTERIOR_STATS)

    # Return the interpolator based on the posterior stats
    try:
        return interpolator_function(
            x_data=df[X_COLUMN_STATS],
            y_data=df[Y_COLUMN_STATS],
            method="linear",
        )
    except Exception as e:
        print(e)
