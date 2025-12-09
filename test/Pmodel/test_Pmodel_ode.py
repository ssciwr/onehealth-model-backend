import numpy as np

from heiplanet_models.Pmodel.Pmodel_ode import rk4_step, albopictus_ode_system


# ---- Pytest Fixtures


# ---- Helper Functions


def negative_ode(state, params):
    # Returns a negative derivative, which would drive state negative
    return -2.0 * state - 1.0


def dummy_log_ode_nonneg(state, params):
    # Returns zeros, so log-correction should keep state at minimum allowed
    return np.zeros_like(state)


def zero_ode(state, params):
    # Returns zero derivative, so state should remain unchanged unless log correction is triggered
    return np.zeros_like(state)


def dummy_log_ode_safe(state, params):
    # Returns a small positive value to test log-correction path
    return np.full_like(state, 1e-6)


def shape_ode(state, params):
    # Returns a vector of ones with the same shape as state
    return np.ones_like(state)


def shape_log_ode(state, params):
    # Returns zeros with the same shape as state
    return np.zeros_like(state)


# ---- Tests RK4 Method
# TODO: add an small integration test (unit test) for rk4_step here, it should solve the ODE dx/dt = a * x with known analytical solution
def test_rk4_step_basic_integration():
    pass


# Test: Negative value correction in rk4_step
def test_rk4_step_negative_value_correction():
    # Initial state is positive
    x0 = np.array([1.0])
    time_step = 0.1
    params = tuple()

    # Run RK4 step with negative ODE
    result = rk4_step(
        ode_func=negative_ode,
        log_ode_func=dummy_log_ode_nonneg,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert output is non-negative (correction applied)
    assert np.all(result >= 0), "rk4_step did not correct negative values"


# Test: Log-ODE path trigger (initial state near zero)
def test_rk4_step_log_ode_path_trigger():
    # Initial state is near zero
    x0 = np.array([1e-30])
    time_step = 0.1
    params = tuple()

    # Run RK4 step with zero ODE and log-ODE correction
    result = rk4_step(
        ode_func=zero_ode,
        log_ode_func=dummy_log_ode_safe,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert output is finite and non-negative
    assert np.all(
        np.isfinite(result)
    ), "rk4_step produced non-finite values for near-zero state"
    assert np.all(result >= 0), "rk4_step produced negative values for near-zero state"


# Test: Shape preservation in rk4_step
def test_rk4_step_shape_preservation():
    # Initial state is a vector of length 5
    x0 = np.ones(5)
    time_step = 0.1
    params = tuple()

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert output shape matches input shape
    assert (
        result.shape == x0.shape
    ), f"rk4_step output shape {result.shape} does not match input {x0.shape}"


# Test: Shape preservation for multidimensional array in rk4_step
def test_rk4_step_shape_preservation_multidim():
    # Initial state is a multidimensional array
    x0 = np.ones((3, 2, 4, 5))
    time_step = 0.1
    params = tuple()

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert output shape matches input shape
    assert (
        result.shape == x0.shape
    ), f"rk4_step output shape {result.shape} does not match input {x0.shape}"


# Test: No side effects (input state is not modified in-place)
def test_rk4_step_no_side_effects():
    x0 = np.ones(5)
    time_step = 0.1
    params = tuple()
    x0_copy = x0.copy()

    _ = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert input state is unchanged
    assert np.array_equal(x0, x0_copy), "rk4_step modified the input state in-place"


# Test: Edge case with zero state (should not produce NaN or negative values)
def test_rk4_step_zero_state():
    x0 = np.zeros(5)
    time_step = 0.1
    params = tuple()

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert output is finite and non-negative
    assert np.all(
        np.isfinite(result)
    ), "rk4_step produced non-finite values for zero state"
    assert np.all(result >= 0), "rk4_step produced negative values for zero state"


# Test: Parameter passing to ODE function
def test_rk4_step_parameter_passing():
    x0 = np.ones(3)
    time_step = 0.1
    test_key = 42
    another_test_key = 100
    params = (test_key, another_test_key)
    called = {}

    def mock_ode(state, model_params):
        called["params"] = model_params
        return np.ones_like(state)

    def mock_log_ode(state, model_params):
        return np.zeros_like(state)

    _ = rk4_step(
        ode_func=mock_ode,
        log_ode_func=mock_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # Assert that model_params was passed to the ODE function
    assert "params" in called, "ODE function was not called with model_params"
    assert (
        called["params"] == params
    ), "ODE function did not receive correct model_params"


# Test: Behavior with NaN values in state
def test_rk4_step_nan_in_state():
    x0 = np.array([1.0, np.nan, 2.0])
    time_step = 0.1
    params = tuple()

    result = rk4_step(
        ode_func=shape_ode,
        log_ode_func=shape_log_ode,
        state=x0,
        model_params=params,
        time_step=time_step,
    )

    # By default, NaNs will propagate unless handled in the ODE function
    assert np.isnan(result[1]), "rk4_step did not propagate NaN as expected"


# ---- Tests for albopictus_ode_system
def test_albopictus_ode_system_shape_preservation():
    # Minimal valid state and parameter arrays (1D, 5 compartments)
    state = np.ones(5)
    # Each parameter must be a scalar or array; use shape (1,) for arrays
    t_idx = 1
    step_t = 1.0
    arr = np.ones(1)
    model_params = (
        t_idx,
        step_t,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        1.0,
        arr,
    )
    result = albopictus_ode_system(state, model_params)
    assert (
        result.shape == state.shape
    ), f"Output shape {result.shape} does not match input {state.shape}"


def test_albopictus_ode_system_positive_inputs_finite_output():
    # All state and parameters positive
    state = np.full(5, 2.0)
    t_idx = 1
    step_t = 1.0
    arr = np.full(1, 2.0)
    model_params = (
        t_idx,
        step_t,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        2.0,
        arr,
    )
    result = albopictus_ode_system(state, model_params)
    assert np.all(
        np.isfinite(result)
    ), "Output contains non-finite values (NaN or inf) for positive inputs"


def test_albopictus_ode_system_zero_state():
    # State is all zeros
    state = np.zeros(5)
    t_idx = 1
    step_t = 1.0
    arr = np.ones(1)
    model_params = (
        t_idx,
        step_t,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        1.0,
        arr,
    )
    result = albopictus_ode_system(state, model_params)
    # Output should be zeros (no population to produce new individuals)
    assert np.all(result == 0), f"Output should be zero, got {result}"


def test_albopictus_ode_system_nan_handling():
    # State contains NaNs
    state = np.array([1.0, np.nan, 2.0, 3.0, 4.0])
    t_idx = 1
    step_t = 1.0
    arr = np.ones(1)
    model_params = (
        t_idx,
        step_t,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        1.0,
        arr,
    )
    result = albopictus_ode_system(state, model_params)
    # Output should have NaNs in the same positions as the input state
    nan_mask = np.isnan(state)
    assert np.all(
        np.isnan(result) == nan_mask
    ), f"NaN propagation mismatch: input {state}, output {result}"


def test_albopictus_ode_system_internal_nan_generation():
    # State has no NaNs, but carrying capacity is zero, which should cause division by zero
    # The function should correct internal NaNs using its NaN-replacement logic
    state = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    t_idx = 1
    step_t = 1.0
    arr = np.ones(1)
    cc = np.zeros(
        1
    )  # carrying capacity zero triggers division by zero in compartment 2
    model_params = (
        t_idx,
        step_t,
        cc,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        arr,
        1.0,
        arr,
    )
    result = albopictus_ode_system(state, model_params)
    # The third compartment (index 2) should be corrected to -state[2] * step_t
    expected = -state[2] * step_t
    assert np.isclose(
        result[2], expected
    ), f"Expected correction to {expected} in compartment 2, got {result}"
    # Output should not contain NaNs
    assert not np.any(np.isnan(result)), f"Output contains NaNs: {result}"


def test_albopictus_ode_system_parameter_unpacking():
    # Use known values for all parameters and state
    state = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    t_idx = 1
    step_t = 0.5
    # All arrays are shape (1,) for broadcasting
    CC = np.array([10.0])
    birth = np.array([1.0])
    dia_lay = np.array([0.2])
    dia_hatch = np.array([0.3])
    mort_e = np.array([0.1])
    mort_j = np.array([0.2])
    mort_a = np.array([0.3])
    ed_surv = np.array([0.4])
    dev_j = np.array([0.5])
    dev_i = np.array([0.6])
    dev_e = 0.7
    water_hatch = np.array([0.8])
    model_params = (
        t_idx,
        step_t,
        CC,
        birth,
        dia_lay,
        dia_hatch,
        mort_e,
        mort_j,
        mort_a,
        ed_surv,
        dev_j,
        dev_i,
        dev_e,
        water_hatch,
    )
    # Manual calculation for each compartment
    # 0: 6 * 1 * (1-0.2) - (0.1 + 0.8*0.7)*2 = 6*0.8 - (0.1+0.56)*2 = 4.8 - 0.66*2 = 4.8 - 1.32 = 3.48
    # 1: 6*1*0.2 - 0.8*0.3*3 = 1.2 - 0.24*3 = 1.2 - 0.72 = 0.48
    # 2: 0.8*0.7*2 + 0.8*0.3*0.4*3 - (0.2+0.5)*4 - (4**2)/10
    #    = 0.56*2 + 0.096*3 - 0.7*4 - 16/10 = 1.12 + 0.288 - 2.8 - 1.6 = 1.408 - 4.4 = -2.992
    # 3: 0.5*4*0.5 - (0.3+0.6)*5 = 2*0.5 - 0.9*5 = 1 - 4.5 = -3.5
    # 4: 0.6*5 - 0.3*6 = 3 - 1.8 = 1.2
    expected = np.array([3.48, 0.48, -2.992, -3.5, 1.2])
    result = albopictus_ode_system(state, model_params)
    assert np.allclose(
        result, expected, atol=1e-6
    ), f"Expected {expected}, got {result}"


# ---- Tests for albopictus_log_ode_system
