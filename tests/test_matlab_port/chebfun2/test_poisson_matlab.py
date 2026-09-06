"""Port of MATLAB Chebfun tests/chebfun2/test_poisson.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_poisson.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2, poisson
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)

pi = np.pi


class TestChebfun2Poisson:
    def test_all_matlab_assertions(self):
        tol = 1e4 * ChebfunPref().cheb2Prefs.chebfun2eps
        v = chebfun2(lambda x, y: (1 - x ** 2) * (1 - y ** 2) * jnp.cos(2 * pi * x * y))
        f = v.lap()
        u = poisson(f)
        assert float((u - v).norm()) < tol                            # pass(1)

        v = chebfun2(lambda x, y: jnp.cos(2 * pi * x * y))
        f = v.lap()
        u = poisson(f, v)
        assert float((u - v).norm()) < tol                            # pass(2)

        m = n = 101
        v = chebfun2(lambda x, y: (1 - x ** 2) * (1 - y ** 2) * jnp.cos(2 * pi * x * y))
        f = v.lap()
        assert float((poisson(f, 0, m, n) - v).norm()) < tol          # pass(3)
        m, n = 87, 101
        assert float((poisson(f, 0, m, n) - v).norm()) < tol          # pass(4)

        m, n = 99, 101
        a, b, c, d = -3, 1, 4, 4.2
        v = chebfun2(lambda x, y: (x - a) * (x - b) * (y - c) * (y - d) * jnp.cos(2 * pi * x * y),
                     domain=(a, b, c, d))
        f = v.lap()
        assert float((poisson(f, 0, m, n) - v).norm()) < tol          # pass(5)

        v = chebfun2(lambda x, y: (x - a) * (x - b) * (y - c) * (y - d) * jnp.cos(2 * pi * (x + y)) + 1,
                     domain=(a, b, c, d))
        f = v.lap()
        assert float((poisson(f, 1, m, n) - v).norm()) < tol          # pass(6)

        m = n = 201
        p = lambda x, y: x * y + jnp.cos(3 * x ** 2 * (y - .2))  # noqa: E731
        v = chebfun2(p, domain=(a, b, c, d))
        f = v.lap()
        assert float((poisson(f, p, m, n) - v).norm()) < tol          # pass(7)

        m, n = 201, 202
        v = chebfun2(p, domain=(a, b, c, d))
        f = v.lap()
        assert float((poisson(f, v, m, n) - v).norm()) < tol          # pass(8)

        u1 = poisson(f, v)
        u2 = poisson(f, v, n)
        u3 = poisson(f, v, m, n)
        u4 = poisson(f, v, n, method="adi")
        u5 = poisson(f, v, n, method="fadi")
        u6 = poisson(f, v, m, n, "adi")
        u7 = poisson(f, v, m, n, "fadi")
        u8 = poisson(f, v, m, n, "bartelsStewart")
        assert (float((u1 - u2).norm()) < tol and float((u2 - u3).norm()) < tol
                and float((u3 - u4).norm()) < tol and float((u4 - u5).norm()) < tol
                and float((u5 - u6).norm()) < tol and float((u6 - u7).norm()) < tol
                and float((u7 - u8).norm()) < tol)                    # pass(9)
