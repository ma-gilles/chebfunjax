"""Port of MATLAB Chebfun tests/chebop2/test_adaptivity.m (Fable 5).

``mldivide(N, f, m, n)`` (``m`` = y-size, ``n`` = x-size, ``inf`` =
adaptive in that direction) is ``N.solve(f, ny=m, nx=n)``; MATLAB ``[n, m] = length(u)`` returns the
column (y) and row (x) lengths.

Provenance
----------
MATLAB source : tests/chebop2/test_adaptivity.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.operators.chebop2 import Chebop2

jax.config.update("jax_enable_x64", True)


def _c2(u):
    return u if isinstance(u, Chebfun2) else Chebfun2(approx=u)


def _len(u):
    n, m = _c2(u).length()
    return int(n), int(m)


class TestChebop2Adaptivity:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        d = (-5, 1, 0, .1)
        N = Chebop2(lambda u: u.diffy() + u.diffx(3), domain=d)
        N.dbc = lambda x: jnp.exp(-10 * x ** 2)
        N.rbc = lambda t, u: [u, u.diff(1)]
        N.lbc = 0
        inf = float("inf")
        u1 = N.solve(0.0, ny=200, nx=inf)
        n, m = _len(u1)
        assert m == 200                                                      # pass(1)
        u2 = N.solve(0.0, nx=200, ny=200)
        m, n = _len(u2)
        assert m == 200                                                      # pass(2)
        assert n == 200                                                      # pass(3)
        u3 = N.solve(0.0, ny=inf, nx=200)
        n, m = _len(u2)
        assert n == 200                                                      # pass(4)
        u4 = N.solve(0.0, nx=10, ny=10)
        n, m = _len(u4)
        assert m == 10 and n == 10                                           # pass(5)-(6)
        u4 = N.solve(0.0, ny=6, nx=10)
        n, m = _len(u4)
        assert n == 10 and m == 6                                            # pass(7)-(8)
        assert float((_c2(u1) - _c2(u2)).norm()) < 20 * np.sqrt(tol)         # pass(9)
        assert float((_c2(u2) - _c2(u3)).norm()) < 20 * np.sqrt(tol)         # pass(10)
