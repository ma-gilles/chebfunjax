"""All ten original test_restrict.m predicates, Chebfun commit 7574c77.

Python .restrict(dom) represents both MATLAB brace indexing and the explicit
subsref({}, dom) call in predicate ten. Construction and assertion order are
unchanged. No sampled norm substitutes are used.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import Domain


def test_native_restrict_ten_predicates():
    tol = 1000*ChebfunPref().cheb3Prefs.chebfun3eps

    def ff(x, y, z):
        return jnp.exp(x/2+y)+jnp.cos(x+z**2)

    d = (-3., -2., -1., 1., 2., 3.)
    f = Chebfun3.from_function(ff, domain=d)
    val = f.restrict((-2.5, -2.5, 0., 0., 2.5, 2.5))
    assert abs(val-ff(-2.5, 0., 2.5)) < tol  # 1
    g = f.restrict((d[0], d[1], 0., 0., 2.5, 2.5))
    exact = Chebfun.from_function(lambda x: ff(x, 0., 2.5), Domain(d[:2]))
    assert (g-exact).norm() < tol  # 2
    g = f.restrict((d[0], d[0], d[2], d[3], d[4], d[4]))
    exact = Chebfun.from_function(lambda y: ff(d[0], y, d[4]), Domain(d[2:4]))
    assert (g-exact).norm() < tol  # 3
    g = f.restrict((d[0], d[0], d[2], d[2], d[4], d[5]))
    exact = Chebfun.from_function(lambda z: ff(d[0], d[2], z), Domain(d[4:6]))
    assert (g-exact).norm() < tol  # 4
    g = f.restrict((d[0], d[0], d[2], d[3], d[4], d[5]))
    exact = Chebfun2.from_function(lambda y, z: ff(d[0], y, z), domain=d[2:6])
    assert (g-exact).norm() < tol  # 5
    g = f.restrict((d[0], d[1], d[2], d[2], d[4], d[5]))
    exact = Chebfun2.from_function(lambda x, z: ff(x, d[2], z), domain=d[:2]+d[4:6])
    assert (g-exact).norm() < tol  # 6
    g = f.restrict((d[0], d[1], d[2], d[3], d[4], d[4]))
    exact = Chebfun2.from_function(lambda x, y: ff(x, y, d[4]), domain=d[:4])
    assert (g-exact).norm() < tol  # 7
    new = (-2.75, -2.25, -.5, .5, 2.25, 2.75)
    g = f.restrict(new)
    exact = Chebfun3.from_function(ff, domain=new)
    assert (g-exact).norm() < tol  # 8
    with pytest.raises(Exception):  # Native catches any exception.
        f.restrict(tuple(x/2 for x in d))  # 9
    val = f.restrict((-2.5, -2.5, 0., 0., 2.5, 2.5))
    assert abs(val-ff(-2.5, 0., 2.5)) < tol  # 10
