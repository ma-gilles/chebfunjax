"""Early branches of @chebtech/diff.m, Chebfun 7574c77.

Copyright 2017 by The University of Oxford and The Chebfun Developers.
Analytical controls; no native random-fixture or full derivative-suite claim.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_empty_returns_before_inspecting_order_or_dimension(cls):
    null = cls.empty()
    assert null.diff(object(), object()) is null
    for shape in [(0,), (0, 3), (3, 0)]:
        f = cls(coeffs=jnp.empty(shape, dtype=jnp.complex128), ishappy=False)
        assert f.diff(object(), object()) is f


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_zero_order_preserves_scalar_and_array_for_any_dimension(cls):
    for shape in [(3,), (3, 2)]:
        f = cls(coeffs=jnp.arange(1, 1 + 3 * (2 if len(shape) == 2 else 1),
                                 dtype=jnp.float64).reshape(shape), ishappy=False)
        for dim in [1, 2, 3, 0]:
            assert f.diff(0, dim) is f


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_none_order_uses_first_derivative_without_changing_happiness(cls):
    # 2+3*T1+4*T2 => 3*T0+16*T1.
    f = cls(coeffs=jnp.array([2., 3., 4.]), ishappy=False)
    df = f.diff(None)
    assert jnp.array_equal(df.coeffs, jnp.array([3., 16.]))
    assert not df.ishappy


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_noncontinuous_dimension_uses_column_differences(cls):
    f = cls(coeffs=jnp.array([[1., 4., 9.], [2., 8., 18.]]), ishappy=False)
    for dim in [0, 2, 3]:
        first = f.diff(None, dim)
        second = f.diff(2, dim)
        assert jnp.array_equal(first.coeffs, jnp.array([[3., 5.], [6., 10.]]))
        assert jnp.array_equal(second.coeffs, jnp.array([[2.], [4.]]))
        assert not first.ishappy and not second.ishappy


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_dispatch_preserves_jit_and_coefficient_derivative(cls):
    coefficients = jnp.array([2., 3., 4.])

    def derivative(c):
        return cls(coeffs=c, ishappy=False).diff(None).coeffs

    assert jnp.array_equal(jax.jit(derivative)(coefficients), jnp.array([3., 16.]))
    assert jnp.array_equal(jax.jacfwd(derivative)(coefficients),
                           jnp.array([[0., 1., 0.], [0., 0., 4.]]))
