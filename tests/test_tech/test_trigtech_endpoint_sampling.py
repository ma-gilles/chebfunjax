"""Source endpoint averaging for callable trigonometric construction.

Provenance
----------
MATLAB source : @trigtech/refine.m, refineResampling
Chebfun commit: 7574c77
The source evaluates [trigpts(n); 1], replaces its first row by the mean
of the two endpoint rows, and drops the last row. Fixed n is the Python
sampling adapter; it must preserve the same endpoint convention. The
adaptive path has no nested-refinement preference API yet.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, _sample_as_trig_dtype, trigpts

EPS = np.finfo(float).eps


@pytest.mark.parametrize('n', [7, 8])
def test_fixed_callable_averages_endpoint_samples(n):
    f = Trigtech.from_function(lambda x: x, n=n)
    expected = np.asarray(trigpts(n)).copy()
    expected[0] = 0.0
    np.testing.assert_allclose(np.asarray(f.values), expected, rtol=0, atol=16*EPS)
    np.testing.assert_allclose(np.asarray(f(jnp.array([-1., 1.]))), 0.,
                               rtol=0, atol=16*EPS)


def test_complex_array_callable_averages_each_column():
    def op(x):
        return jnp.stack([x, (2-3j)*x, x*x], axis=-1)
    f = Trigtech.from_function(op, n=8)
    expected = np.asarray(op(trigpts(8))).copy()
    expected[0] = [0, 0, 1]
    assert not f.is_real
    np.testing.assert_allclose(np.asarray(f.values), expected, rtol=0, atol=64*EPS)


def test_unhappy_adaptive_step_keeps_averaged_endpoints():
    op = lambda x: .5*(1+jnp.sign(x))  # noqa: E731
    with pytest.warns(UserWarning, match='did not converge with 16 points'):
        f = Trigtech.from_function(op, maxpow2=4)
    assert not f.ishappy
    expected = np.asarray(op(trigpts(16))).copy()
    expected[0] = .5
    np.testing.assert_allclose(np.asarray(f.values), expected, rtol=0, atol=16*EPS)
    np.testing.assert_allclose(np.asarray(f(jnp.array([-1., 1.]))), .5,
                               rtol=0, atol=16*EPS)


def test_fixed_sampling_batches_include_right_endpoint():
    batches = []
    def op(x):
        batches.append(np.asarray(x).copy())
        return x
    Trigtech.from_function(op, n=8)
    expected = np.concatenate([np.asarray(trigpts(8)), [1.]])
    np.testing.assert_array_equal(batches[-1], expected)


def test_from_values_preserves_user_endpoint_data():
    values = np.arange(8, dtype=float)
    f = Trigtech.from_values(jnp.asarray(values))
    np.testing.assert_allclose(np.asarray(f.values), values, rtol=0, atol=16*EPS)


def test_offgrid_sampling_keeps_both_endpoint_values():
    points = jnp.asarray([-1., -.125, 1.])
    actual, is_real = _sample_as_trig_dtype(lambda x: x, points)
    assert is_real
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(points))
