"""Original assertions from Chebfun tests/chebfun3/test_min3.m, 7574c77.

Original functions, construction order, norms and thresholds are retained.
Python None represents the native empty second argument for max2/min2.
Passing these predicates does not establish full extrema algorithm parity.
"""

from __future__ import annotations

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref


def test_native_min3():
    tol = 1e4*ChebfunPref().cheb3Prefs.chebfun3eps
    a, b, c = 0.3, -0.4322, -0.83343
    f = chebfun3(lambda x, y, z: (x-a)**2+(y-b)**2+(z-c)**2)
    exact = 0
    value, _ = f.min3()
    assert abs(value-exact) < tol
