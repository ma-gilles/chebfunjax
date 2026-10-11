"""Native NE guard ordering and valid complement control, Chebfun7574c77."""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


def test_empty_identity_precedes_array_guard():
    empty = cj.chebfun()
    array = cj.chebfun(lambda x: jnp.stack((x, x), axis=-1))
    assert empty.logical_ne(array) is empty
    assert array.logical_ne(empty) is empty


def test_numerical_complement_branch_unchanged():
    f = cj.chebfun(jnp.sin, domain=[-1.0, 0.0, 1.0])
    g = cj.chebfun(lambda x: 0 * x + 0.5)
    result = f.logical_ne(g)
    previous = 1 - f.logical_eq(g)
    assert result.isequal(previous)
    np.testing.assert_array_equal(result.point_values, previous.point_values)
