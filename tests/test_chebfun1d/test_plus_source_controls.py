"""Independent addition controls supplement the source test's vacuous errors.

Provenance
----------
MATLAB source : @chebfun/plus.m, @chebfun/thresholdBreakpointValues.m,
    @chebtech/plus.m
Chebfun commit: 7574c77
"""
import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


def test_transposed_scalar_and_array_addition_preserves_orientation():
    f = cj.chebfun(lambda x: 1.+2j*x).T
    points = jnp.array([-.75, -.125, .25, .625])
    result = f+jnp.array([[1.], [2.], [3.]])
    assert result.is_transposed
    assert result(points).shape == (3, 4)
    expected = (1.+2j*points)[None, :]+jnp.array([1., 2., 3.])[:, None]
    assert jnp.max(jnp.abs(result(points)-expected)) < 32*jnp.finfo(jnp.float64).eps
    assert (f+(2.+3j)).is_transposed
    assert (f+f).is_transposed
    array = cj.chebfun(lambda x: jnp.stack((x, x*x), axis=-1))
    object.__setattr__(array, '_point_values', jnp.array([[3., 5.], [7., 11.]]))
    assert jnp.array_equal((array.T+array.T)(jnp.array([-1., 1.])),
                            jnp.array([[6., 14.], [10., 22.]]))


def test_unsigned_operand_really_raises_source_leaf_error():
    f = cj.chebfun(jnp.sin)
    with pytest.raises(TypeError, match='CHEBFUN:CHEBTECH:plus:typeMismatch'):
        _ = f+jnp.uint8(128)


def test_orientation_mismatch_really_raises_source_error():
    f = cj.chebfun(jnp.sin)
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:plus:matdim'):
        _ = f+f.T


def test_expanded_stored_point_values_and_source_threshold():
    f = cj.chebfun(lambda x: x).define_point(0., 5.)
    result = f+[1., 2., 3.]
    assert jnp.array_equal(result(jnp.array([-1., 0., 1.])),
                            jnp.array([[0., 1., 2.], [6., 7., 8.], [2., 3., 4.]]))
    small = cj.chebfun(lambda x: x).define_point(0., 1e-16)
    assert (small+0.)(0.) == 0.
    assert abs((small+0.)(.25)-.25) < 1e-15


def test_quasimatrix_numeric_columns_and_empty():
    f = cj.chebfun(lambda x: x)
    q = Quasimatrix([f, f*f, f+1.], f.domain)
    result = q+[1., 2., 3.]
    assert all(c.n_columns == 1 for c in result.cols)
    assert jnp.max(jnp.abs(result(.5)-jnp.array([1.5, 2.25, 4.5]))) < 1e-15
    assert (q+[]).isempty()
    assert (q+cj.chebfun()).isempty()
    with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN:plus:dims'):
        _ = q+[1., 2.]


def test_scalar_addition_jit_and_derivative():
    f = cj.chebfun(lambda x: 1.+x)
    x = jnp.array([-1., -.2, .7, 1.])
    evaluate = jax.jit(lambda a: (f+a)(x))
    assert jnp.max(jnp.abs(evaluate(.25)-(1.25+x))) < 2e-15
    assert jnp.max(jnp.abs(jax.jacfwd(evaluate)(.25)-1.)) < 2e-15
