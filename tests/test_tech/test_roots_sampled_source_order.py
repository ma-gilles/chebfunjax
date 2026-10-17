"""Dynamic sampled subdivision preserves the current source Clenshaw order.

Provenance: @chebtech/roots.m and @chebtech/clenshaw.m, Chebfun7574c77.
These controls exercise odd/even remainders with one shared capacity and
independent source-provider execution, using unchanged raw-bit comparisons.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import _roots_sampled_cpu as helper
from chebfunjax.tech.chebtech import _clenshaw, _vals2coeffs_jax


@pytest.mark.parametrize("n", [3, 4, 17, 18, 39, 40])
def test_dynamic_states_and_finish_match_source_provider(n):
    capacity = 64
    coeffs = np.zeros(capacity)
    coeffs[:n] = np.random.default_rng(7574+n).normal(size=n)
    # Mix ordinary and cancellation-sensitive magnitudes without fitting to
    # outputs. All active points are real and remain within the source interval.
    coeffs[1:n:3] *= 1e-8
    points = jnp.linspace(-1., 1., 2*capacity)
    c = jnp.asarray(coeffs)
    b1, b2, finite = helper.recurrence_states(c, points, jnp.asarray(n))
    assert bool(finite)
    actual = helper.finish_subdivision(c, points, b1, b2, n)
    values = _clenshaw(c[:n], points[:2*n])
    expected = _vals2coeffs_jax(values[:n]), _vals2coeffs_jax(values[n:])
    for child, original in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(child).view(np.uint64),
                                      np.asarray(original).view(np.uint64))
