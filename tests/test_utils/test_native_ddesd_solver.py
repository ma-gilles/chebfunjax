"""Added analytic controls for the source DDESD constant-delay subset."""

import jax.numpy as jnp
import pytest

from chebfunjax.utils.native_ddesd import ddesd


def test_vector_history_and_argument_forwarding():
    def fun(t, y, z, factor):
        return factor * z[:, 0]

    def history(t, factor):
        return jnp.array([1.0, 2.0])

    sol = ddesd(fun, [2.0], history, [0.0, 0.5], {"RelTol": 1e-8, "AbsTol": 1e-10}, 2.0)
    assert sol.success and sol.x[0] == 0.0 and sol.x[-1] == 0.5
    assert sol.nsteps == len(sol.x) - 1
    expected = jnp.array([1.0, 2.0])[:, None] * (1 + 2 * sol.x[None, :])
    assert float(jnp.max(jnp.abs(sol.y - expected))) < 100 * jnp.finfo(float).eps
    assert jnp.array_equal(sol.yp, jnp.broadcast_to(jnp.array([2.0, 4.0])[:, None], sol.yp.shape))


def test_real_delayed_polynomial_segments():
    def fun(t, y, z):
        return z[:, 0]

    sol = ddesd(fun, [1.0], [1.0], [0.0, 2.0], {"RelTol": 1e-7, "AbsTol": 1e-10})
    expected = 1 + sol.x + jnp.maximum(sol.x - 1, 0) ** 2 / 2
    assert sol.success and sol.x[-1] == 2.0
    # Independent added global-error control, not a translated native bound.
    assert float(jnp.max(jnp.abs(sol.y[0] - expected))) < 5e-6
    assert sol.nfailed > 0


def test_negative_start_and_explicit_fixed_step():
    def fun(t, y, z):
        return jnp.ones_like(y)

    sol = ddesd(fun, [1.0], [0.0], [-1.0, 0.0], {"MinStep": 0.25, "MaxStep": 0.25, "RelTol": 1e-6})
    assert sol.success and sol.nsteps == 4 and sol.nfailed == 0
    assert float(jnp.max(jnp.abs(sol.y[0] - (sol.x + 1)))) < 20 * jnp.finfo(float).eps


@pytest.mark.parametrize("delays", [[0.0], [-1.0], []])
def test_nonpositive_or_empty_delay_rejected(delays):
    with pytest.raises(NotImplementedError):
        ddesd(lambda t, y, z: y, delays, [1.0], [0.0, 1.0])
