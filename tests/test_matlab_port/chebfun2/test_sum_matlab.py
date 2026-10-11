"""Literal seven MATLAB Chebfun sum predicates.

Provenance
----------
MATLAB source : tests/chebfun2/test_sum.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp

from chebfunjax import chebfun, chebfun2


def test_native_seven_sum_predicates():
    eps = float(jnp.finfo(jnp.float64).eps)
    tol = 100 * eps
    f = chebfun2(lambda x,y: x**2 + 4*y, domain=(11.,14.,7.,10.))
    assert abs(float(f.sum2()) - 1719) < 30 * tol
    f = chebfun2(lambda x,y: x)
    assert float((f.sum() - chebfun(lambda x:2*x).T).norm()) < tol
    assert float((f.sum(1) - f.sum()).norm()) < 10*eps
    assert float(f.sum(2).norm()) < 10*eps
    f = chebfun2(lambda x,y:y,domain=(0.,1.,-jnp.pi,jnp.pi))
    assert float(f.sum().norm()) < 10*eps
    assert float((f.sum(1)-f.sum()).norm()) < tol
    assert float((f.sum(2)-chebfun(lambda x:x,domain=(-jnp.pi,jnp.pi))).norm()) < tol
