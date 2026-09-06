"""Port of MATLAB Chebfun tests/chebop2/test_backwardsWaveEquation.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_backwardsWaveEquation.m
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


class TestChebop2BackwardsWaveEquation:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        d = (-pi, pi, 0, 1)
        exact = chebfun2(lambda x, t: jnp.sin(x + t), domain=d)
        N = Chebop2(lambda u: u.diff(2, 0) - u.diff(0, 2), domain=d)
        N.lbc = lambda t: jnp.sin(-pi + t)
        N.rbc = lambda t: jnp.sin(pi + t)
        N.ubc = lambda x, u: [u - jnp.sin(x + 1), u.diff(1) - jnp.cos(x + 1)]
        u = N.solve(0.0)
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        if not isinstance(u, Chebfun2):
            u = Chebfun2(approx=u)
        assert float((u - exact).norm()) < 10 * tol                          # pass(1)
