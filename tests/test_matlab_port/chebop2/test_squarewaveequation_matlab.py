"""Port of MATLAB Chebfun tests/chebop2/test_squarewaveequation.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_squarewaveequation.m
Chebfun commit: 7574c77

MATLAB ``diff(u, k, 1)`` (k-th derivative in y = t) is the proxy's
``u.diff(k, 0)``; ``diff(u, k, 2)`` is ``u.diff(0, k)``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.operators.chebop2 import Chebop2

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _err(u, exact):
    from chebfunjax.chebfun2d.chebfun2 import Chebfun2
    if not isinstance(u, Chebfun2):
        u = Chebfun2(approx=u)
    return float((u - exact).norm())


class TestChebop2Squarewaveequation:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        d = (0, 10, 0, 10)
        exact = chebfun2(lambda x, t: jnp.sin(x + t) + jnp.sin(x - t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(t) + jnp.sin(-t)
        N.rbc = lambda t: jnp.sin(10 + t) + jnp.sin(10 - t)
        N.dbc = lambda x, u: [u - 2 * jnp.sin(x), u.diff(1)]
        assert _err(N.solve(0.0), exact) < 150 * tol                         # pass(1)

        d = (0, 1, 0, 1)
        exact = chebfun2(lambda x, t: jnp.sin(x + t) + jnp.cos(x - t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(t) + jnp.cos(-t)
        N.rbc = lambda t: jnp.sin(1 + t) + jnp.cos(1 - t)
        N.dbc = lambda x, u: [u - jnp.sin(x) - jnp.cos(x), u.diff(1) - jnp.cos(x) - jnp.sin(x)]
        assert _err(N.solve(0.0), exact) < tol                               # pass(2)

        d = (0, pi, 0, pi)
        exact = chebfun2(lambda x, t: jnp.sin(x + t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(t)
        N.rbc = lambda t: jnp.sin(pi + t)
        N.dbc = lambda x, u: [u - jnp.sin(x), u.diff(1) - jnp.cos(x)]
        err = _err(N.solve(0.0), exact)
        assert err < 5 * tol                                                 # pass(3)
