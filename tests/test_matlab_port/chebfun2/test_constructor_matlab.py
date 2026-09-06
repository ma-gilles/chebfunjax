"""Port of MATLAB Chebfun tests/chebfun2/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_constructor.m
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


class TestChebfun2Constructor:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = lambda x, y: jnp.cos(x) + jnp.sin(x * y)  # noqa: E731
        fstr = "cos(x) + sin(x.*y)"
        f1 = chebfun2(f)
        f2 = chebfun2(f, domain=(-1, 1, -1, 1))
        assert float((f1 - f2).norm()) < tol                            # pass(1)
        f3 = chebfun2(fstr)
        f4 = chebfun2(fstr, domain=(-1, 1, -1, 1))
        assert float((f1 - f3).norm()) < tol                            # pass(2)
        assert float((f3 - f4).norm()) < tol                            # pass(3)
        f5 = chebfun2("2")
        f6 = chebfun2(lambda x, y: 2)
        assert float((f5 - f6).norm()) < tol and float((f5 - 2).norm()) == 0  # pass(4)

        g = lambda x, y: jnp.cos(x) * y + x * jnp.sin(y)  # noqa: E731
        f = chebfun2(g)
        assert f.rank == 2                                               # pass(5): length(f)
        assert abs(float(f(.1, np.pi / 6)) - float(g(.1, np.pi / 6))) < tol  # pass(6)

        f = lambda x, y: 1.0 / (1 + 25 * x ** 2 * y ** 2)  # noqa: E731
        ffch = chebfun2(lambda x, y: f(x, y), domain=(-2, 2, -2, 2))
        xx = np.linspace(-2, 2, 50)
        XX, YY = np.meshgrid(xx, xx)
        err = np.max(np.abs(np.asarray(f(jnp.asarray(XX), jnp.asarray(YY)))
                            - np.asarray(ffch(jnp.asarray(XX), jnp.asarray(YY)))))
        assert err < 2e4 * tol                                           # pass(7)

        f = chebfun2(lambda x, y: 1, vectorize=True)
        g = chebfun2(1)
        assert float((f - g).norm()) < tol                               # pass(8)
        assert float((f - g).norm()) < tol                               # pass(9)

        f = chebfun2(lambda x, y: jnp.sin(np.pi * x) * jnp.cos(np.pi * y), trig=True)  # p.tech = @trigtech
        g = chebfun2(lambda x, y: jnp.sin(np.pi * x) * jnp.cos(np.pi * y),
                     domain=(-1, 1, -1, 1), trig=True)
        assert float((f - g).norm()) < tol                               # pass(10)

        f = chebfun2(lambda x, y: jnp.sin(80 * x + y))
        assert len(f.cols.cols[0]) < 50                                  # pass(11)

        f = chebfun(lambda x: 1.0 / (1 + 25 * x ** 2))
        f2 = chebfun2(lambda x, y: 1.0 / (1 + 25 * x ** 2))
        assert len(f2.rows.cols[0]) < len(f) + 20                        # pass(12)

        f = chebfun2(lambda x, y: jnp.cos(np.pi * x * y))
        frows = chebfun(f.rows.cols[0])                                  # chebfun(f.rows)
        assert abs(len(frows) - len(f.rows.cols[0])) < 10                # pass(13)
