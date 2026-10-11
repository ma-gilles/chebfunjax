"""Rational evaluator shape, complex-query and support contracts.

Provenance
----------
MATLAB source : minimax.m computeTrialFunctionRational and reval.
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.minimax import _make_reval


@pytest.mark.parametrize('shape', [(), (5,), (1, 5)])
@pytest.mark.parametrize('complex_query', [False, True])
def test_linear_rational_shape_and_complex_query(shape, complex_query):
    # These support weights represent r(z)=z identically.
    r = _make_reval(jnp.array([-1., 0., 1.]),
                    jnp.array([.5, 0., -.5]), jnp.array([.5, -1., .5]))
    x = jnp.asarray(.3) if not shape else jnp.linspace(-.7, .7, 5).reshape(shape)
    if complex_query:
        x = x + .25j
    for result in [r(x), jax.jit(r)(x)]:
        assert result.shape == x.shape
        np.testing.assert_allclose(result, x, rtol=1e-13, atol=1e-14)


def test_support_values_and_nan_input():
    r = _make_reval(jnp.array([-1., 0., 1.]),
                    jnp.array([.5, 0., -.5]), jnp.array([.5, -1., .5]))
    values = r(jnp.array([-1., 0., 1., jnp.nan]))
    np.testing.assert_array_equal(values[:3], [-1., 0., 1.])
    assert jnp.isnan(values[3])


def test_pole_outside_support_is_not_repaired():
    # r(z)=1/(1-2z), so z=.5 is a genuine pole, not a support singularity.
    r = _make_reval(jnp.array([0., 1.]), jnp.array([-1., 1.]), jnp.ones(2))
    np.testing.assert_array_equal(r(jnp.array([0., 1.])), [1., -1.])
    assert jnp.isposinf(r(.5))
