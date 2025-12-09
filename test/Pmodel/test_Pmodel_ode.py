import numpy as np
import xarray as xr
import pytest

from heiplanet_models.Pmodel.Pmodel_ode import (
    rk4_step,
    albopictus_ode_system,
    albopictus_log_ode_system,
    call_function,
)


@pytest.fixture
def call_function_test_arrays():
    coords = {
        "longitude": [0, 1],
        "latitude": [10, 20],
        "time": [0, 1, 2, 3],
    }
    temperature = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    temperature_mean = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    latitudes = xr.DataArray([10, 20], dims=["latitude"])
    carrying_capacity = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    egg_activate = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    time_step = 1.0
    return (
        coords,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )


@pytest.fixture
def call_function_initial_state():
    return np.ones((2, 2, 5))


@pytest.fixture
def call_function_random_state():
    return np.random.rand(2, 2, 5)


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
    params = ()

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
    params = ()

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
    params = ()

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
    params = ()

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
    params = ()
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
    params = ()

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
    params = ()

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
    result = albopictus_ode_system(state, model_params)
    # Manual calculation notes for regression testing:
    # For the given input, the expected output is:
    # [3.48, 0.48, -2.992, -3.5, 1.2]
    # These values were obtained by running the function with the same inputs.
    expected = np.array([3.48, 0.48, -2.992, -3.5, 1.2])
    assert np.allclose(
        result, expected, atol=1e-6
    ), f"Expected {expected}, got {result}"


# ---- Tests for albopictus_log_ode_system
def test_albopictus_log_ode_system_shape_preservation():
    # Minimal valid state and parameter arrays (1D, 5 compartments)
    state = np.ones(5)
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
    result = albopictus_log_ode_system(state, model_params)
    assert (
        result.shape == state.shape
    ), f"Output shape {result.shape} does not match input {state.shape}"


def test_albopictus_log_ode_system_positive_inputs_finite_output():
    # All state and parameters positive and nonzero
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
    result = albopictus_log_ode_system(state, model_params)
    assert np.all(
        np.isfinite(result)
    ), "Output contains non-finite values (NaN or inf) for positive inputs"


def test_albopictus_log_ode_system_zero_state():
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
    result = albopictus_log_ode_system(state, model_params)
    # Output should be zeros (NaNs corrected to -state * step_t, which is zero)
    assert np.all(result == 0), f"Output should be zero, got {result}"


def test_albopictus_log_ode_system_nan_handling():
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
    result = albopictus_log_ode_system(state, model_params)
    # Output should have NaNs in the same positions as the input state
    nan_mask = np.isnan(state)
    assert np.all(
        np.isnan(result) == nan_mask
    ), f"NaN propagation mismatch: input {state}, output {result}"


def test_albopictus_log_ode_system_internal_nan_correction():
    # Internal NaN generation (e.g., 0/0 division) should be corrected to finite output
    state = np.zeros(5)
    t_idx = 1
    step_t = 1.0
    arr = np.zeros(1)
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
    result = albopictus_log_ode_system(state, model_params)
    assert np.all(
        np.isfinite(result)
    ), "Output contains non-finite values after internal NaN correction"


def test_albopictus_log_ode_system_parameter_unpacking():
    # Use known values for all parameters and state
    state = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    t_idx = 1
    step_t = 0.5
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
    result = albopictus_log_ode_system(state, model_params)
    # Manual calculation summary for log ODE system:
    # 0: (6*1*(1-0.2))/2 - (0.1+0.8*0.7) = 1.74
    # 1: (6*1*0.2)/3 - 0.8*0.3 = 0.16
    # 2: 0.8*0.7*2/4 + 0.8*0.3*0.4*3/4 - (0.2+0.5) - 4/10 = -0.748
    # 3: 0.5*0.5*4/5 - (0.3+0.6) = -0.7
    # 4: 0.6*5/6 - 0.3 = 0.2
    expected = np.array([1.74, 0.16, -0.748, -0.7, 0.2])
    assert np.allclose(
        result, expected, atol=1e-6
    ), f"Expected {expected}, got {result}"


def test_albopictus_log_ode_system_negative_state_correction():
    # Test that negative values in state are corrected to finite output
    state = np.array([-1.0, -2.0, -3.0, -4.0, -5.0])
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
    result = albopictus_log_ode_system(state, model_params)
    # Output should be finite (no NaNs or infs)
    assert np.all(
        np.isfinite(result)
    ), "Output contains non-finite values for negative state"


# ---- Tests for call_function
def test_call_function_shape_preservation():

    # Minimal 2x2 grid, 4 time steps, 5 compartments
    state = np.ones((2, 2, 5))
    coords = {
        "longitude": [0, 1],
        "latitude": [10, 20],
        "time": [0, 1, 2, 3],
    }
    temperature = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    temperature_mean = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    latitudes = xr.DataArray([10, 20], dims=["latitude"])
    carrying_capacity = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    egg_activate = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    time_step = 1.0

    result = call_function(
        state,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )
    # Output shape: (2, 2, 5, 4)
    assert result.shape == (
        2,
        2,
        5,
        4,
    ), f"call_function output shape {result.shape} is incorrect"


# Test: Initial state propagation in call_function
def test_call_function_initial_state_propagation():
    # Minimal 2x2 grid, 4 time steps, 5 compartments
    state = np.random.rand(2, 2, 5)
    coords = {
        "longitude": [0, 1],
        "latitude": [10, 20],
        "time": [0, 1, 2, 3],
    }
    temperature = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    temperature_mean = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    latitudes = xr.DataArray([10, 20], dims=["latitude"])
    carrying_capacity = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    egg_activate = xr.DataArray(
        np.ones((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    time_step = 1.0

    result = call_function(
        state.copy(),
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )
    # The first time step in result should be close to the initial state (allowing for model logic)
    # This is a minimal check: the model may update state in the first step, but shape and type must match
    assert (
        result[..., 0].shape == state.shape
    ), "First time step shape does not match initial state"
    assert np.all(
        np.isfinite(result[..., 0])
    ), "First time step contains non-finite values"


# Test: Integration progression in call_function
def test_call_function_integration_progression(
    call_function_test_arrays, call_function_initial_state
):
    (
        _,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    ) = call_function_test_arrays
    state = call_function_initial_state
    result = call_function(
        state.copy(),
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )
    # Check that at least one time step's state differs from the initial state
    initial_state = state
    n_time = result.shape[-1]
    changed = False
    for t in range(n_time):
        if not np.allclose(result[:, :, :, t], initial_state):
            changed = True
            break
    assert (
        changed
    ), "No time step state differs from initial state; integration may not be progressing"


# Test: Zero state propagation in call_function
def test_call_function_zero_state(call_function_test_arrays):
    (
        coords,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    ) = call_function_test_arrays
    # All inputs and initial state are zeros
    state = np.zeros((2, 2, 5))
    temperature = xr.DataArray(
        np.zeros((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    temperature_mean = xr.DataArray(
        np.zeros((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    carrying_capacity = xr.DataArray(
        np.zeros((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    egg_activate = xr.DataArray(
        np.zeros((2, 2, 4)), dims=["longitude", "latitude", "time"], coords=coords
    )
    # latitudes can remain as in the fixture
    result = call_function(
        state,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )
    # Output should be all zeros
    assert np.all(
        result == 0
    ), f"call_function output should be all zeros, got {result}"


# Test: Single time step edge case in call_function
def test_call_function_single_time_step():
    coords = {
        "longitude": [0, 1],
        "latitude": [10, 20],
        "time": [0],
    }
    state = np.ones((2, 2, 5))
    temperature = xr.DataArray(
        np.ones((2, 2, 1)), dims=["longitude", "latitude", "time"], coords=coords
    )
    temperature_mean = xr.DataArray(
        np.ones((2, 2, 1)), dims=["longitude", "latitude", "time"], coords=coords
    )
    latitudes = xr.DataArray([10, 20], dims=["latitude"])
    carrying_capacity = xr.DataArray(
        np.ones((2, 2, 1)), dims=["longitude", "latitude", "time"], coords=coords
    )
    egg_activate = xr.DataArray(
        np.ones((2, 2, 1)), dims=["longitude", "latitude", "time"], coords=coords
    )
    time_step = 1.0

    result = call_function(
        state,
        temperature,
        temperature_mean,
        latitudes,
        carrying_capacity,
        egg_activate,
        time_step,
    )
    # Output shape should be (2, 2, 5, 1)
    assert result.shape == (
        2,
        2,
        5,
        1,
    ), f"call_function output shape {result.shape} is incorrect for single time step"
    assert np.all(
        np.isfinite(result)
    ), "call_function output contains non-finite values for single time step"
