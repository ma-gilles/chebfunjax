"""Port of MATLAB Chebfun tests/chebfun2/test_trig.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_trig.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

D2R = np.pi / 180.0


class TestChebfun2Trig:
    def test_all_matlab_assertions(self):
        tol = 100 * ChebfunPref().cheb2Prefs.chebfun2eps
        fns = [("cos", jnp.cos), ("sin", jnp.sin), ("tan", jnp.tan),
               ("cosh", jnp.cosh), ("sinh", jnp.sinh), ("tanh", jnp.tanh),
               ("tand", lambda t: jnp.tan(D2R * t))]
        g = chebfun2(lambda x, y: x * y ** 2)
        for jj, (name, fn) in enumerate(fns):
            h = getattr(g, name)()
            exact = chebfun2(lambda x, y, _fn=fn: _fn(x * y ** 2))
            assert float((h - exact).norm()) < tol, name                # pass(1)-(7)

        rng = np.random.RandomState(0)
        u = chebfun(jnp.asarray(rng.rand(10)), trig=True)
        v = chebfun(jnp.asarray(rng.rand(10)), trig=True)
        exact = chebfun2(lambda x, y: u(y) * v(x), trig=True)           # u * v.'
        coeffs = jnp.outer(u.trigcoeffs(), v.trigcoeffs())
        f = chebfun2(coeffs, coeffs=True, trig=True)
        assert float((f - exact).norm()) < tol                          # pass(8)

        C0 = np.zeros((3, 3))
        C = C0.copy()
        C[1, 1] = 1
        f = chebfun2(jnp.asarray(C), trig=True, coeffs=True)
        assert float((f - 1).norm()) < tol                              # pass(9)

        x = chebfun2(lambda x, y: x)
        y = chebfun2(lambda x, y: y)
        C = C0.copy()
        C[0, 1] = 1
        f = chebfun2(jnp.asarray(C), trig=True, coeffs=True)
        assert float((f - (-1j * np.pi * y).exp()).norm()) < tol        # pass(10)
        C = C0.copy()
        C[1, 0] = .5
        C[1, 2] = .5
        f = chebfun2(jnp.asarray(C), trig=True, coeffs=True)
        assert float((f - (np.pi * x).cos()).norm()) < tol              # pass(11)

        f = chebfun2(lambda x, y: jnp.sin(2 * np.pi * x), trig=True)
        g = chebfun2(lambda x, y: jnp.cos(2 * np.pi * x) + 2, trig=True)
        assert (f + 1).isPeriodicTech()                                 # pass(12)
        assert (f ** 2).isPeriodicTech()                                # pass(13)
        assert (f / g).isPeriodicTech()                                 # pass(14)
