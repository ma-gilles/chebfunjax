"""Port of MATLAB Chebfun tests/chebop2/test_waveequation.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_waveequation.m
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


class TestChebop2Waveequation:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        d = (-pi, pi, 0, 1)
        exact = chebfun2(lambda x, t: jnp.sin(x + t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(-pi + t)
        N.rbc = lambda t: jnp.sin(pi + t)
        N.dbc = lambda x, u: [u - jnp.sin(x), u.diff(1) - jnp.cos(x)]
        assert _err(N.solve(0.0), exact) < 5 * tol                           # pass(1)

        exact = chebfun2(lambda x, t: jnp.sin(x + t) + jnp.sin(x - t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = 0
        N.rbc = 0
        N.dbc = lambda x, u: [u - 2 * jnp.sin(x), u.diff(1)]
        assert _err(N.solve(0.0), exact) < 2 * tol                           # pass(2)

        d = (-2 * pi, 2 * pi, 0, 1)
        c = 2
        exact = chebfun2(lambda x, t: jnp.sin(x + c * t) + jnp.sin(x - c * t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - c ** 2 * u.diff(0, 2), domain=d)
        N.lbc = 0
        N.rbc = 0
        N.dbc = lambda x, u: [u - 2 * jnp.sin(x), u.diff(1)]
        assert _err(N.solve(0.0), exact) < 50 * tol                          # pass(3)

        c = 30
        exact = chebfun2(lambda x, t: jnp.sin(x + c * t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - c ** 2 * u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(-2 * pi + c * t)
        N.rbc = lambda t: jnp.sin(2 * pi + c * t)
        N.dbc = lambda x, u: [u - jnp.sin(x), u.diff(1) - c * jnp.cos(x)]
        assert _err(N.solve(0.0), exact) < 7e3 * tol                         # pass(4)

        d = (-2 * pi, 2 * pi, 1, 2)
        c = 3
        exact = chebfun2(lambda x, t: jnp.sin(x + c * t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - c ** 2 * u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(-2 * pi + c * t)
        N.rbc = lambda t: jnp.sin(2 * pi + c * t)
        N.dbc = lambda x, u: [u - jnp.sin(c + x), u.diff(1) - c * jnp.cos(x + c)]
        assert _err(N.solve(0.0), exact) < 40 * tol                          # pass(5)
