import numpy as np


MIN_TEMP_CELSIUS = 0
MAX_TEMP_CELSIUS = 45


def R0_pip_fun(temperature, interpolator):

    # Converts the variable temperature into 1D-Numpy array
    Tm = np.asarray(temperature)

    # Creates an array with the same dimensions of Tm, full of NaN values
    result = np.full_like(Tm, np.nan, dtype=float)

    mask_valid_values = (
        ~np.isnan(Tm) & (Tm >= MIN_TEMP_CELSIUS) & (Tm <= MAX_TEMP_CELSIUS)
    )

    result[mask_valid_values] = interpolator(Tm[mask_valid_values])

    return result


# x_new = [0, 1, -1, 8, 8.1, 8.3]
# y_new = R0_pip_fun(x_new)

# for x, y in zip(x_new, y_new):
#     print(f"{x} -> {y}")
