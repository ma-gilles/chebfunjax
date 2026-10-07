"""All twelve pinned partition/combine clauses, without tolerance inflation.

Provenance
----------
MATLAB source : tests/spherefun/test_partitionCombine.m
Chebfun commit: 7574c77
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun


def _sph(fun):
    return Spherefun.from_function(lambda lam, th: fun(
        jnp.cos(lam)*jnp.sin(th), jnp.sin(lam)*jnp.sin(th), jnp.cos(th)))


def test_all_twelve_source_clauses():
    tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
    even, odd = Spherefun.empty().partition()
    assert even.isempty() and odd.isempty()  # 1
    assert Spherefun.combine(Spherefun.empty(), Spherefun.empty()).isempty()  # 2
    def fe(x, y, z):
        return jnp.sin(jnp.pi*x*y)

    def fo(x, y, z):
        return jnp.sin(jnp.pi*x*z)
    f = _sph(fe)
    even, odd = f.partition()
    assert even.isequal(f)  # 3
    assert odd.isempty()  # 4
    f = _sph(fo)
    even, odd = f.partition()
    assert odd.isequal(f)  # 5
    assert even.isempty()  # 6
    f = _sph(lambda x, y, z: fe(x,y,z)+fo(x,y,z))
    even, odd = f.partition()
    assert float((_sph(fe)-even).norm()) < tol  # 7
    assert float((_sph(fo)-odd).norm()) < tol  # 8
    even = _sph(fe)
    f = Spherefun.combine(even, Spherefun.empty())
    assert float((even-f).norm()) < tol  # 9
    odd = _sph(fo)
    f = Spherefun.combine(Spherefun.empty(), odd)
    assert float((odd-f).norm()) < tol  # 10
    reference = _sph(lambda x,y,z: fe(x,y,z)+fo(x,y,z))
    f = Spherefun.combine(_sph(fe), _sph(fo))
    assert float((reference-f).norm()) < tol  # 11
    f = _sph(lambda x,y,z: fe(x,y,z)+fo(x,y,z))
    with pytest.raises(ValueError, match="CHEBFUN:SPHEREFUN:combine:parity"):
        Spherefun.combine(f,f)  # 12
