from Pmodel_initial import load_data
from Pmodel_functions import (
    carrying_capacity,
    water_hatching,
)
from Pmodel_params import (
    CONSTANTS_CARRYING_CAPACITY,
    CONSTANTS_WATER_HATCHING,
)


def print_slices(dataset, value):
    for i in range(value):  # for indices 0, 1, 2
        print(f"Slice at time index {i}:")
        print(dataset.isel(time=i).values)
        print()  # Blank line for readability


if __name__ == "__main__":

    # ---- Load data from datalake golden zone (for now local)
    model_data = load_data(time_step=10)
    print(model_data.print_attributes())

    # TODO: create function carrying capacity
    # ---- Calculate Carrying Capacity
    constants_dummy_cc = {
        "ALPHA_RAIN": 10.0,
        "ALPHA_DENS": 10.0,
        "GAMMA": 10.0,
        "LAMBDA": 10.0,
    }
    CC = carrying_capacity(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
        constants=constants_dummy_cc,
    )

    print_slices(CC, 3)

    # TODO: create function water hatching
    # ---- Calculate Water Hatching
    egg_active = water_hatching(
        rainfall_data=model_data.rainfall,
        population_data=model_data.population_density,
        constants=CONSTANTS_WATER_HATCHING,
    )

    print_slices(egg_active, 3)

    # TODO: Create function to solve the ODEs
