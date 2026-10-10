"""Original assertions from Chebfun tests/chebfun3/test_cumsum3.m, 7574c77.

Original functions, construction order, norms and thresholds are retained.
Python None represents the native empty second argument for max2/min2.
Passing these predicates does not establish full extrema algorithm parity.
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_cumsum3():
    tol = 100*ChebfunPref().cheb3Prefs.chebfun3eps
    x = chebfun3(lambda x, y, z: x)
    y = chebfun3(lambda x, y, z: y)
    z = chebfun3(lambda x, y, z: z)
    f = x
    g = f.cumsum3()
    exact = (y+1)*(z+1)*(x**2/2-1/2)
    assert (g-exact).norm() < tol
