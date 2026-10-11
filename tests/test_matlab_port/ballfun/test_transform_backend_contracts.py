"""JAX execution and empty contracts for native ballfun transforms.

Provenance: @ballfun/coeffs2vals.m and vals2coeffs.m, Chebfun7574c77.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun


@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4)])
def test_empty_identity(shape):
    value = jnp.empty(shape, dtype=jnp.float64)
    for transform in (Ballfun.vals2coeffs, Ballfun.coeffs2vals):
        for run in (transform, jax.jit(transform)):
            result = run(value)
            assert result.shape == value.shape
            assert result.dtype == value.dtype


def test_complex_scalar_jit():
    value = jnp.asarray(3+4j)
    for transform in (Ballfun.vals2coeffs, Ballfun.coeffs2vals):
        result = jax.jit(transform)(value)
        assert result.shape == ()
        assert result == value


def test_complex_mode_jit():
    # T2(r)*exp(i*lambda)*exp(-i*theta), odd radial/Fourier dimensions.
    r = jnp.cos(jnp.arange(4, -1, -1)*jnp.pi/4)
    lam = -jnp.pi + 2*jnp.pi*jnp.arange(3)/3
    th = -jnp.pi + 2*jnp.pi*jnp.arange(5)/5
    values = (2*r*r-1)[:, None, None]*jnp.exp(1j*lam)[None,:,None]*jnp.exp(-1j*th)[None,None,:]
    expected = jnp.zeros((5,3,5),dtype=jnp.complex128).at[2,2,1].set(1)
    actual = jax.jit(Ballfun.vals2coeffs)(values)
    assert jnp.max(jnp.abs(actual-expected)) < 100*jnp.finfo(jnp.float64).eps
    restored = jax.jit(Ballfun.coeffs2vals)(expected)
    assert jnp.max(jnp.abs(restored-values)) < 100*jnp.finfo(jnp.float64).eps
