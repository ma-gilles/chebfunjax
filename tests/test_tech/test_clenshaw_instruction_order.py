"""Preserve the previous ordered-operation IEEE behavior during loop consolidation.

The reference retains the six independently ordered binary calls from 9888b5d5.
Native finite values and independent JVP/VJP oracles live in the adjacent tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _clenshaw
from chebfunjax.utils._ordered_binary import binary


@jax.jit
def _previous_ordered_operations(coeffs, x):
    doubled = binary(jnp.asarray(2., x.dtype), x, 0)
    degree = coeffs.shape[0] - 1

    def step(i, state):
        first, second = state
        k = degree - 2*i
        second = binary(binary(coeffs[k], binary(doubled, first, 0), 1), second, 2)
        first = binary(binary(coeffs[k-1], binary(doubled, second, 0), 1), first, 2)
        return first, second

    zero = jnp.zeros_like(x)
    first, second = jax.lax.fori_loop(0, degree//2, step, (zero, zero))
    if degree % 2:
        first, second = binary(binary(coeffs[1], binary(doubled, first, 0), 1), second, 2), first
    half = binary(jnp.asarray(.5, x.dtype), doubled, 0)
    return binary(binary(coeffs[0], binary(half, first, 0), 1), second, 2)


@pytest.mark.parametrize("nonfinite_x", [False, True], ids=["finite_x", "nonfinite_x"])
@pytest.mark.parametrize("coeffs", [
    [0., -0., 0., -0.],
    [1e308, 1e308, -1e308, 1e308],
    [1., np.inf, -np.inf, np.nan],
    [1e-310, -1e-310, 1e-310, -1e-310],
    [1., -2., 3., -4., 5.],
])
def test_ieee_results_match_separately_ordered_operations(coeffs, nonfinite_x):
    # Values outside [-1,1] exercise extrapolation and intermediate overflow;
    # no assumption about finite recurrence states may hide those cases.
    points = [-1e308, -2., -1., -0., 0., 1., 2., 1e308]
    if nonfinite_x:
        points += [np.inf, -np.inf, np.nan]
    x = jnp.asarray(points)
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    expected = np.asarray(_previous_ordered_operations(coeffs, x))
    actual = np.asarray(_clenshaw(coeffs, x))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    non_nan = ~np.isnan(expected)
    # Includes signed zeros, subnormals and signed infinities. NaN payloads
    # are not a public JAX numerical contract.
    np.testing.assert_array_equal(actual[non_nan].view(np.uint64),
                                  expected[non_nan].view(np.uint64))
