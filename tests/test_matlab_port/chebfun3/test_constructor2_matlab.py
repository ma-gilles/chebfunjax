"""Port of MATLAB Chebfun tests/chebfun3/test_constructor2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_constructor2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _n(f):
    return float(f.norm())


class TestChebfun3Constructor2:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps
        rng = np.random.RandomState(0)
        for sz in (3, 4):
            exact = rng.rand(sz, sz, sz)
            C = np.asarray(chebfun3(jnp.asarray(exact), coeffs=True).chebcoeffs3())
            assert np.linalg.norm(exact.ravel() - C.ravel()) < 10 * tol      # pass(1)-(2)

        dom = (-1, 1, -1, 1, -1, 1)
        u = lambda x, y, z: jnp.cos(pi * x) * jnp.sin(pi * y) * jnp.cos(pi * z)  # noqa: E731
        assert _n(chebfun3(u, trig=True) - chebfun3(u, dom, trig=True)) < tol  # pass(3)
        v = lambda x, y, z: jnp.cos(pi * jnp.cos(pi * x) + pi * jnp.sin(pi * y) + pi * jnp.cos(pi * z))  # noqa: E731
        f1 = chebfun3(v, trig=True)
        f2 = chebfun3(v, dom, trig=True)
        assert _n(f1 - f2) < 10 * tol                                        # pass(4)
        assert isinstance(f1.cols[0], Trigtech)                              # pass(5)
        assert isinstance(f1.rows[0], Trigtech)                              # pass(6)
        assert isinstance(f1.tubes[0], Trigtech)                             # pass(7)
        f1 = chebfun3(v, periodic=True)
        f2 = chebfun3(v, dom, periodic=True)
        assert _n(f1 - f2) < 10 * tol                                        # pass(8)
        assert isinstance(f1.cols[0], Trigtech)                              # pass(9)
        assert isinstance(f1.rows[0], Trigtech)                              # pass(10)
        assert isinstance(f1.tubes[0], Trigtech)                             # pass(11)

        f = chebfun3(1, coeffs=True)
        assert _n(f - 1) < tol                                               # pass(12)

        ff = lambda x, y, z: -x * jnp.sin(jnp.sqrt(x + 15 + 2 * y + 3 * z))  # noqa: E731
        f = chebfun3(ff)
        fEps = chebfun3(ff, eps=1e-8)
        m, n, p = f.length()
        mE, nE, pE = fEps.length()
        assert mE < m and nE < n and pE < p                                  # pass(13)
        r1, r2, r3 = f.rank
        r1E, r2E, r3E = fEps.rank
        assert r1E < r1 and r2E < r2 and r3E < r3                            # pass(14)

        f = chebfun3("pi")
        assert abs(float(f(0, 0, 0)) - pi) < tol                             # pass(15)
        f = chebfun3("cos(alpha)")
        assert abs(float(f(0, 0, 0)) - np.cos(0)) < tol                      # pass(16)
        f = chebfun3("x+y")
        assert abs(float(f(0.25, 0.5, 0)) - 0.75) < tol                      # pass(17)
        f = chebfun3("cos(x+y+z)")
        assert abs(float(f(0.25, 0.5, -0.3)) - np.cos(0.25 + 0.5 - 0.3)) < tol  # pass(18)

        with pytest.raises(ValueError, match="CHEBFUN:CHEBFUN3:constructor:equi"):
            chebfun3(lambda x, y, z: z * x ** 2 * jnp.exp(jnp.sin(x + y)), equi=True)  # pass(19)
