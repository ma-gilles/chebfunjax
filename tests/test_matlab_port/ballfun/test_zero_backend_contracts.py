"""Exact native nnz-based zero semantics under JAX execution.

Provenance: @ballfun/iszero.m, Chebfun7574c77.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.ballfun.ballfun import Ballfun


@pytest.mark.parametrize("value,expected", [(0.,True),(1e-20,False),(1e-20j,False),(float('nan'),False)])
def test_iszero_jit(value, expected):
    f = Ballfun(jnp.asarray(value).reshape((1,1,1)), is_real=False, domain=(0.,1.,-jnp.pi,jnp.pi,0.,jnp.pi))
    assert bool(jax.jit(lambda x: x.iszero())(f)) is expected


def test_empty_iszero_jit():
    assert bool(jax.jit(lambda f: f.iszero())(Ballfun.empty()))
