"""Port of MATLAB Chebfun tests/chebop2/test_withoutAD.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_withoutAD.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.operators.chebop2 import Chebop2

jax.config.update("jax_enable_x64", True)


def _c2(u):
    return u if isinstance(u, Chebfun2) else Chebfun2(approx=u)


class TestChebop2WithoutAD:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        N = Chebop2()
        N.domain = (-1, 1, -1, 1)
        N.U = [[500, 1], [0, 0], [1, 0]]
        N.S = [[1, 0], [0, 1]]
        N.V = [[1, 500], [0, 0], [0, 1]]
        N.lbc = 1
        N.rbc = 1
        N.ubc = 1
        N.dbc = 1
        N.xorder = 2
        N.yorder = 2
        u = _c2(N.solve(0.0))
        N = Chebop2(lambda u: u.diffx(2) + u.diffy(2) + 1000 * u)
        N.lbc = 1
        N.rbc = 1
        N.ubc = 1
        N.dbc = 1
        exact = _c2(N.solve(0.0))
        x = np.linspace(-1, 1, 1000)
        xx, yy = np.meshgrid(x, x)
        A = np.abs(np.asarray(u(jnp.asarray(xx), jnp.asarray(yy))) - np.asarray(exact(jnp.asarray(xx), jnp.asarray(yy))))
        assert np.max(A) < 20000 * tol                                       # pass(1)

        xc = chebfun(lambda x: x)
        exact = chebfun2(lambda x, y: 3 * y ** 2 + x ** 3)
        N = Chebop2()
        N.domain = (-1, 1, -1, 1)
        N.U = [[0, 1], [0, 0], [1, 0]]
        N.S = [[1, 0], [0, 1]]
        N.V = [[-xc, 0], [0, 0], [0, 1]]
        N.lbc = exact(-1.0, ":")
        N.rbc = exact(1.0, ":")
        N.ubc = exact(":", 1.0)
        N.dbc = exact(":", -1.0)
        N.xorder = 2
        N.yorder = 2
        u = _c2(N.solve(0.0))
        A = np.abs(np.asarray(u(jnp.asarray(xx), jnp.asarray(yy))) - np.asarray(exact(jnp.asarray(xx), jnp.asarray(yy))))
        assert np.max(A) < tol                                               # pass(2)
