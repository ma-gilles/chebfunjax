"""Eager JAX scalar multiplication/division against analytic and source contracts.

MATLAB source: @separableApprox/mtimes.m, @separableApprox/rdivide.m
Chebfun commit: 7574c77
The native scalar predicate is numel(g)==1. Dynamic traced scalars are outside
this eager adapter; existing static-scalar JIT/AD controls remain separate.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


@pytest.fixture(scope="module")
def f():
    return Chebfun2.from_function(lambda x, y: (1+x)*(2-y))


@pytest.mark.parametrize("shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("value", [2.5, -3., 2+1j, 0.])
def test_eager_jax_singleton_scalar_arithmetic(f, shape, value):
    scalar = jnp.asarray(value).reshape(shape)
    x = jnp.asarray([-.75, -.2, .4, .9])
    y = jnp.asarray([.7, -.3, .1, -.8])
    analytic = (1+x)*(2-y)
    for result in (scalar*f, f*scalar):
        np.testing.assert_allclose(result(x, y), value*analytic, rtol=0, atol=2e-13)
        if value == 0:
            assert result.rank == 1
            assert bool(jnp.all(jnp.isinf(result.pivot_values)))
    if value:
        result = f/scalar
        np.testing.assert_allclose(result(x, y), analytic/value, rtol=0, atol=2e-13)
    else:
        with pytest.raises(ZeroDivisionError):
            f/scalar


@pytest.mark.parametrize("shape", [(0,), (2,), (1, 2)])
def test_nonscalar_jax_arrays_not_treated_as_scalar(f, shape):
    values = jnp.ones(shape)
    for operation in (lambda: f*values, lambda: values*f, lambda: f/values):
        with pytest.raises(TypeError):
            operation()


def test_float32_operand_keeps_represented_scalar_value(f):
    scalar = jnp.asarray(1.234567, dtype=jnp.float32)
    expected = float(scalar)*3.
    for result in (scalar*f, f*scalar):
        np.testing.assert_allclose(result(.5, 0.), expected, rtol=0, atol=2e-13)
