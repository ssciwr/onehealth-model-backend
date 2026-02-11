from dataclasses import dataclass

import numpy as np


@dataclass
class PmodelOutput:
    model_output: np.ndarray


def create_model_output(initial_conditions, temperature_shape, time_step):

    number_longitudes, number_latitudes, number_ode_variables = initial_conditions.shape

    number_times = int(temperature_shape[-1] / time_step)

    shape_output = (
        number_longitudes,
        number_latitudes,
        number_ode_variables,
        number_times,
    )

    print(shape_output)

    empty_model_output = np.zeros(shape=shape_output, dtype=np.float64)

    return empty_model_output
