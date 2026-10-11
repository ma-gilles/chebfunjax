"""Independent source-equation controls; no captured native values in kernels."""

import jax.numpy as jnp
import pytest

from chebfunjax.utils.native_ddesd import _hermite, _lag_values, _next_step, _trial, ddesd


def history(t):
    return jnp.array([99.0])


def unit_rhs(t, y, z):
    return jnp.ones_like(y)


def test_cubic_hermite_values_and_derivatives():
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y, yp = _hermite(
            t, 0.0, jnp.array([0.0, 2.0]), 1.0, jnp.array([1.0, 3.0]), jnp.zeros(2), jnp.ones(2) * 3
        )
        assert jnp.array_equal(y, jnp.array([t**3, 2 + t**3]))
        assert jnp.array_equal(yp, jnp.ones(2) * 3 * t * t)


def test_history_order_endpoints_and_unused_capacity():
    times = jnp.array([0.0, 1.0, 2.0, 0.0, 0.0])
    values = jnp.array([[7.0, 1.0, 8.0, 0.0, 0.0]])
    slopes = jnp.array([[0.0, 3.0, 12.0, 0.0, 0.0]])
    z, implicit = _lag_values(
        history, (), 2.0, jnp.array([3.0, 2.0, 1.0, 0.5]), times, values, slopes, 3
    )
    assert jnp.array_equal(z, jnp.array([[99.0, 7.0, 1.0, 3.375]]))
    assert implicit


@pytest.mark.parametrize(
    "accepted,nofailed,expected",
    [(True, True, 2.0), (True, False, 1.0), (False, True, 0.5), (False, False, 0.5)],
)
def test_step_controller(accepted, nofailed, expected):
    ratio = 1 / 256 if accepted else 16.0
    actual = _next_step(1.0, ratio, 1.0, nofailed, accepted, 0.0, 10.0)
    assert float(actual) == expected


@pytest.mark.parametrize("delay,calls", [(1.0, 6), (0.1, 10)])
def test_rk4_trial_and_implicit_iteration(delay, calls):
    times = jnp.zeros(4)
    values = jnp.zeros((1, 4))
    slopes = jnp.zeros((1, 4)).at[0, 0].set(1.0)
    result = _trial(
        unit_rhs,
        history,
        (),
        jnp.array([delay]),
        0.0,
        jnp.array([0.0]),
        jnp.array([1.0]),
        0.5,
        0.5,
        True,
        jnp.array([1.0]),
        1e-3,
        True,
        times,
        values,
        slopes,
        1,
        jnp.array([0.0]),
        jnp.array([1.0]),
    )
    assert abs(float(result[0][0]) - 0.5) < 8 * jnp.finfo(float).eps
    assert float(result[1][0]) == 1.0
    assert float(result[2]) < 8 * jnp.finfo(float).eps
    assert int(result[3]) == calls


@pytest.mark.parametrize("kind", ["events", "function_delays", "continuation", "normcontrol"])
def test_unsupported_paths_rejected(kind):
    def delays(t, y):
        return jnp.array([t - 1.0])

    if kind != "function_delays":
        delays = [1.0]
    options = (
        {"Events": lambda t, y, z: 1.0}
        if kind == "events"
        else ({"NormControl": "on"} if kind == "normcontrol" else {})
    )
    hist = {"x": [0.0], "y": [1.0]} if kind == "continuation" else [1.0]
    with pytest.raises(NotImplementedError):
        ddesd(unit_rhs, delays, hist, [0.0, 1.0], options)


def test_rejected_implicit_trial_retains_predictor_state():
    # Accepted history is exact for y'=y(t-.02), history1, through t=.01.
    # The next oversized implicit attempt genuinely fails the source bound.
    def rhs(t, y, z):
        return z[:, 0]

    def past(t):
        return jnp.ones(1)

    times = jnp.array([0.0, 0.01, 0.0, 0.0])
    values = jnp.array([[1.0, 1.01, 0.0, 0.0]])
    slopes = jnp.array([[1.0, 1.0, 0.0, 0.0]])
    y = jnp.array([1.01])
    yp = jnp.ones(1)
    rtol = 1e-8
    first = _trial(
        rhs,
        past,
        (),
        jnp.array([0.02]),
        0.01,
        y,
        yp,
        0.1,
        0.11,
        False,
        jnp.ones(1),
        rtol,
        True,
        times,
        values,
        slopes,
        2,
        y,
        yp,
    )
    assert float(first[2]) > rtol and int(first[3]) == 10
    h = _next_step(0.1, first[2], rtol, True, False, 0.0, 1.0)
    assert float(h) == 0.05 and h > 0.02
    retry = _trial(
        rhs,
        past,
        (),
        jnp.array([0.02]),
        0.01,
        y,
        yp,
        h,
        0.01 + h,
        False,
        jnp.ones(1),
        rtol,
        False,
        first[4],
        first[5],
        first[6],
        2,
        first[0],
        first[1],
    )
    reset = _trial(
        rhs,
        past,
        (),
        jnp.array([0.02]),
        0.01,
        y,
        yp,
        h,
        0.01 + h,
        False,
        jnp.ones(1),
        rtol,
        False,
        first[4],
        first[5],
        first[6],
        2,
        y,
        yp,
    )
    assert float(jnp.max(jnp.abs(retry[0] - reset[0]))) > 100 * jnp.finfo(float).eps
    assert int(retry[3]) >= 9
