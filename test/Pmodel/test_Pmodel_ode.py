import numpy as np

from heiplanet_models.Pmodel.Pmodel_ode import rk4_step


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
    params = test_key
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


# ---- Tests for eqsys function


# ---- Tests for log_eqsys function
