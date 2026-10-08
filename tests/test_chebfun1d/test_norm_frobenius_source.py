"""Source default/Frobenius and explicit spectral norms for array Chebfuns.

Provenance
----------
MATLAB source: @chebfun/norm.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import chebfun


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('complex_values', [False, True])
def test_array_default_and_spectral_are_distinct_scalars(row, complex_values):
    factor = 1+1j if complex_values else 1.
    f = chebfun(lambda x: jnp.stack((1+0*x, factor*x), axis=-1))
    if row:
        f = f.transpose()
    energy = 2+abs(factor)**2*2/3
    for actual in [f.norm(), f.norm('fro')]:
        assert jnp.ndim(actual) == 0
        assert abs(float(actual)-energy**.5) < 1e-13
    assert abs(float(f.norm(2))-max(2., abs(factor)**2*2/3)**.5) < 1e-13


@pytest.mark.parametrize('row', [False, True])
def test_correlated_columns_and_scalar_frobenius(row):
    f = chebfun(lambda x: jnp.stack((1+x, 2*(1+x)), axis=-1))
    scalar = chebfun(lambda x: 1+x)
    if row:
        f, scalar = f.transpose(), scalar.transpose()
    expected = (5*8/3)**.5
    assert abs(float(f.norm())-expected) < 1e-13
    assert abs(float(f.norm(2))-expected) < 1e-13
    assert abs(float(scalar.norm('fro'))-(8/3)**.5) < 1e-13
