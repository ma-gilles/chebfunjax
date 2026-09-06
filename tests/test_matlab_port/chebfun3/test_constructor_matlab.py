"""Port of MATLAB Chebfun tests/chebfun3/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _n(f):
    return float(f.norm())


class TestChebfun3Constructor:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb3Prefs.chebfun3eps
        ff = lambda x, y, z: jnp.cos(x + z) + jnp.sin(x * y * z)  # noqa: E731
        fstr = "cos(x+z) + sin(x.*y.*z)"
        f1 = chebfun3(ff)
        f2 = chebfun3(ff, (-1, 1, -1, 1, -1, 1))
        assert _n(f1 - f2) < tol                                             # pass(1)
        f3 = chebfun3(fstr)
        f4 = chebfun3(fstr, (-1, 1, -1, 1, -1, 1))
        assert _n(f1 - f3) < tol                                             # pass(2)
        assert _n(f3 - f4) < tol                                             # pass(3)
        f5 = chebfun3(lambda x, y, z: f4(x, y, z))
        assert _n(f3 - f5) < tol                                             # pass(4)
        for k in (1, 2, 3):
            f5 = chebfun3(lambda x, y, z: f4(x, y, z), fiberDim=k)
            assert _n(f3 - f5) < tol                                         # pass(5)-(7)

        g = lambda x, y, z: jnp.cos(x) * y * z + x * z * jnp.sin(y)  # noqa: E731
        f = chebfun3(g)
        assert abs(float(f(.1, pi / 6, -0.3)) - float(g(.1, pi / 6, -0.3))) < tol  # pass(8)

        g = lambda x, y, z: x * y * z + y ** 2 * z  # noqa: E731
        f = chebfun3(g)
        r1, r2, r3 = f.rank
        assert r1 == 2 and r2 == 2 and r3 == 1                               # pass(9)-(11)

        f = lambda x, y, z: 1.0 / (1 + x ** 2 * y ** 2 * z ** 2)  # noqa: E731
        ffch = chebfun3(lambda x, y, z: f(x, y, z), (-2, 2, -2, 2, -2, 2))
        xx = np.linspace(-2, 2, 100)
        XX, YY, ZZ = np.meshgrid(xx, xx, xx)
        X, Y, Z = (jnp.asarray(a.ravel()) for a in (XX, YY, ZZ))
        assert np.max(np.abs(np.asarray(f(X, Y, Z)) - np.asarray(ffch(X, Y, Z)))) < 1e5 * tol  # pass(12)

        f = chebfun3(lambda x, y, z: 1, vectorize=True)
        g = chebfun3(1)
        assert _n(f - g) < tol                                               # pass(13)
        f = chebfun3(lambda x, y, z: 1, vectorize=True)
        g = chebfun3(1)
        assert _n(f - g) < tol                                               # pass(14)

        f = chebfun3(lambda x, y, z: jnp.sin(80 * x + y + z))
        _m, n, p = f.length()
        assert n < 50                                                        # pass(15)
        assert p < 50                                                        # pass(16)

        f = chebfun(lambda x: 1.0 / (1 + 25 * x ** 2))
        f2 = chebfun3(lambda x, y, z: 1.0 / (1 + 25 * x ** 2))
        assert f2.length()[1] < len(f) + 20                                  # pass(17)
