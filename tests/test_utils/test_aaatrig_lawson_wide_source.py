"""Wide native svd(A,0) keeps full V; Chebfun aaatrig.m,7574c77."""
import importlib

import jax.numpy as jnp

module = importlib.import_module('chebfunjax.utils.aaa')


def test_wide_lawson_selects_native_nullspace():
    # Three data points, two supports -> literal3x4 Lawson matrix.
    # Its unique nullspace is span([4,-2,-5,1]), which fits [2,3,5].
    # Reduced V has only three rows and omits this required direction.
    z = jnp.array([0., .5, 1.])
    values = jnp.array([2., 3., 5.])
    cauchy = 1/jnp.sin((z[:, None]-jnp.array([0., 1.])[None, :])/2)
    scale = 1/jnp.sin(.25)
    matrix = jnp.array([[2., 4., 0., 0.],
                        [scale, 3*scale, -scale, -3*scale],
                        [0., 0., 2., 10.]])
    coefficients, _, error = module._trig_lawson_step_source(
        matrix, cauchy, values, jnp.array([0, 2]), jnp.ones(3))
    expected = jnp.array([4., -2., -5., 1.])/jnp.sqrt(46.)
    assert abs(abs(jnp.vdot(expected, coefficients))-1) < 1e-14
    assert jnp.linalg.norm(matrix@coefficients) < 1e-13
    assert error < 1e-13
