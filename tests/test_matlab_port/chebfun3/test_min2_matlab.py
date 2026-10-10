"""Original assertions from Chebfun tests/chebfun3/test_min2.m, 7574c77.

Original functions, construction order, norms and thresholds are retained.
Python None represents the native empty second argument for max2/min2.
Passing these predicates does not establish full extrema algorithm parity.
"""

from __future__ import annotations

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_min2():
    tol = 1e12*ChebfunPref().cheb3Prefs.chebfun3eps
    f = chebfun3(lambda x, y, z: x**2+y**2+z**2)
    g = chebfun(lambda z: z**2)
    h1 = f.min2()
    h2 = f.min2(None)
    h3 = f.min2(None, (1, 2))
    h4 = f.min2(None, (2, 1))
    h5 = f.min2(None, (1, 3))
    h6 = f.min2(None, (2, 3))

    assert (h1-g).norm() < tol
    assert (h2-g).norm() < tol
    assert (h3-g).norm() < tol
    assert (h4-g).norm() < tol
    assert (h5-g).norm() < tol
    assert (h6-g).norm() < tol
