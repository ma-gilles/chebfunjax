"""Source exhausted derivative branches, @chebtech/diff.m at7574c77.

Copyright 2017 by The University of Oxford and The Chebfun Developers.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech import chebtech as module
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('shape', [(1,), (3,), (3, 2)])
def test_exhausted_derivative_is_fresh_real_resolved_zero(cls, shape, monkeypatch):
    coefficients = jnp.full(shape, 2. + 3j, dtype=jnp.complex128)
    f = cls(coeffs=coefficients, ishappy=False)

    def forbidden(*args):
        raise AssertionError('Source exhausted derivative must bypass recurrence')

    monkeypatch.setattr(module, '_diff_coeffs', forbidden)
    for order in [shape[0], shape[0] + 2, 100000]:
        df = f.diff(order)
        assert df is not f and df.ishappy
        assert df.coeffs.shape == (1,) + shape[1:]
        assert df.coeffs.dtype == jnp.float64
        assert jnp.array_equal(df.coeffs, jnp.zeros_like(df.coeffs))
    assert not f.ishappy and jnp.array_equal(f.coeffs, coefficients)


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_high_order_zero_works_in_jit_and_has_zero_jacobian(cls):
    c = jnp.array([1., 2., 3.])

    def exhausted(values):
        return cls(coeffs=values, ishappy=False).diff(3).coeffs

    assert jnp.array_equal(jax.jit(exhausted)(c), jnp.zeros((1,)))
    assert jnp.array_equal(jax.jacfwd(exhausted)(c), jnp.zeros((1, 3)))


@pytest.mark.parametrize('cls', [Chebtech1, Chebtech2])
def test_below_length_retains_happiness_and_exact_polynomial(cls):
    # D^2(2+3T1+4T2)=16, but k=2<n=3 preserves original happiness.
    f = cls(coeffs=jnp.array([2., 3., 4.]), ishappy=False)
    df = f.diff(2)
    assert jnp.array_equal(df.coeffs, jnp.array([16.]))
    assert not df.ishappy
