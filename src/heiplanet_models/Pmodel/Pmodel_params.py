# -----------------------------------------------
# ----      General function constants       ----
# -----------------------------------------------

# Ref: https://doi.org/10.1016/0304-3800(94)00034-F
CONSTANTS_REVOLUTION_ANGLE = {
    "CONST_1": 0.2163108,
    "CONST_2": 0.9671396,
    "CONST_3": 0.0086,
    "CONST_4": 367,
    "CONST_5": 186,
}

# Ref: https://doi.org/10.1016/0304-3800(94)00034-F
CONSTANT_DECLINATION_ANGLE = 0.39795

# ----------------------------------------------
# ----      Birth function constants        ----
# ----------------------------------------------
CONSTANTS_MOSQUITO_BIRTH = {
    "CONST_1": 38.8,
    "CONST_2": 33.2,
    "CONST_3": -0.5,
    "CONST_4": 70.3,
    "CONST_5": 14.1,
    "CONST_6": 1.5,
}

CONSTANTS_MOSQUITO_DIAPAUSE_LAY = {
    "RATIO_DIA_LAY": 0.5,
    "CONST_1": 10.058,
    "CONST_2": 0.08965,
    "DAYLENGTH_COEFFICIENT": 0,
}

CONSTANTS_MOSQUITO_DIAPAUSE_HATCHING = {
    "PERIOD": 7,
    "CPP": 11.25,
    "CTT": 11.0,
    "RATIO_DIA_HATCH": 0.1,
    "DAYLENGTH_COEFFICIENT": 0,
}

CONSTANTS_WATER_HATCHING = {
    "E_OPT": 8.0,
    "E_VAR": 0.05,
    "E_0": 1.5,
    "E_RAT": 0.2,
    "E_DENS": 0.01,
    "E_FAC": 0.01,
}
