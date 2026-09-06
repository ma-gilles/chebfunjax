"""Port of MATLAB Chebfun tests/chebfun2/test_integralEqns.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_integralEqns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import chebfun2, fred
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


def _n(f):
    return float(f.norm(2))


class TestChebfun2IntegralEqns:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        s1 = np.sin(1)
        F = chebfun2(lambda x, y: jnp.cos(x) + jnp.sin(y))
        v = chebfun(jnp.cos)
        g = F.fred(v)
        exact = chebfun(lambda x: 2 * s1 * jnp.cos(x)).T
        assert _n(g - exact) < tol                                    # pass(1)
        g = F.T.fred(v)
        exact = chebfun(lambda x: 2 * s1 * jnp.sin(x) + 1 + s1 * np.cos(1)).T
        assert _n(g - exact) < tol                                    # pass(2)

        F = chebfun2(lambda x, y: jnp.cos(x) + jnp.sin(y), domain=(-3, 4, -2, 0))
        v = chebfun(jnp.cos, domain=(-2.0, 0.0))
        g = F.fred(v)
        exact = chebfun(lambda x: np.sin(2) * jnp.cos(x) - np.sin(2) ** 2 / 2,
                        domain=(-3.0, 4.0)).T
        assert _n(g - exact) < tol                                    # pass(3)
        v = chebfun(jnp.cos, domain=(-3.0, 4.0))
        g = F.T.fred(v)
        exact = chebfun(lambda x: (np.sin(3) + np.sin(4)) * jnp.sin(x)
                        + .25 * (14 + np.sin(6) + np.sin(8)), domain=(-2.0, 0.0)).T
        assert _n(g - exact) < tol                                    # pass(4)

        F = chebfun2(lambda x, y: jnp.cos(x) + jnp.sin(y))
        v = chebfun(jnp.cos)
        g = F.volt(v)
        exact = chebfun(lambda x: .25 * (np.cos(2) - jnp.cos(2 * x))
                        + (jnp.sin(x) + s1) * jnp.cos(x)).T
        assert _n(g - exact) < tol                                    # pass(5)
        g = F.T.volt(v)
        exact = chebfun(lambda x: jnp.sin(x) * (jnp.sin(x) + s1)
                        + .25 * (2 * x + jnp.sin(2 * x) + 2 + np.sin(2))).T
        assert _n(g - exact) < tol                                    # pass(6)

        F = fred(lambda s, t: jnp.exp(-jnp.abs(s - t)), (-1.0, 1.0))  # fred(K, domain)
        assert [float(v) for v in F.domain] == [-1.0, 1.0]          # pass(7)
