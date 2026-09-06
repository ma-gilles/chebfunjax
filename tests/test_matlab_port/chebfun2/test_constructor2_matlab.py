"""Port of MATLAB Chebfun tests/chebfun2/test_constructor2.m (Fable 5).

MATLAB's ``chebfun2(f, [m n])`` / ``chebfun2(f, r)`` forms are
``chebfun2(f, n=(m, n))`` and ``chebfun2(f, rank=r)``.

Provenance
----------
MATLAB source : tests/chebfun2/test_constructor2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.trigtech import Trigtech

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _len2(f):
    m, n = f.length()
    return int(m), int(n)


class TestChebfun2Constructor2:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        rng = np.random.RandomState(0)
        r = rng.rand(3, 3)
        assert np.linalg.norm(r - np.asarray(chebfun2(jnp.asarray(r), coeffs=True).chebcoeffs2())) < 10 * tol  # pass(1)
        r = rng.rand(4, 4)
        assert np.linalg.norm(r - np.asarray(chebfun2(jnp.asarray(r), coeffs=True).chebcoeffs2())) < 10 * tol  # pass(2)
        r = rng.rand(4, 4)
        assert np.linalg.norm(r - np.asarray(chebfun2(jnp.asarray(r)).chebpolyval2())) < 10 * tol  # pass(3)

        g = lambda x, y: jnp.exp(-((x + pi) ** 2 + y ** 2) / jnp.maximum(1 - ((x + pi) ** 2 + y ** 2), 0))  # noqa: E731
        f = chebfun2(g, domain=(-pi, pi, -pi, pi))
        xx, yy = np.meshgrid(np.linspace(-pi, pi, 1001), np.linspace(-pi, pi, 1001))
        err = np.asarray(g(jnp.asarray(xx), jnp.asarray(yy))) - np.asarray(f(jnp.asarray(xx), jnp.asarray(yy)))
        assert np.max(np.abs(err)) < 2e7 * tol                          # pass(4)

        f1 = chebfun2(lambda x, y: jnp.cos(pi * x) * jnp.sin(pi * y), trig=True)
        f2 = chebfun2(lambda x, y: jnp.cos(pi * x) * jnp.sin(pi * y), domain=(-1, 1, -1, 1), trig=True)
        assert float((f1 - f2).norm()) < tol                             # pass(5)
        f1 = chebfun2(lambda x, y: jnp.cos(pi * jnp.cos(pi * x) + pi * jnp.sin(pi * y)), trig=True)
        f2 = chebfun2(lambda x, y: jnp.cos(pi * jnp.cos(pi * x) + pi * jnp.sin(pi * y)),
                      domain=(-1, 1, -1, 1), trig=True)
        assert float((f1 - f2).norm()) < 10 * tol                        # pass(6)
        assert isinstance(f1.cols.cols[0].funs[0].tech, Trigtech)        # pass(7)
        assert isinstance(f1.rows.cols[0].funs[0].tech, Trigtech)        # pass(8)
        f1 = chebfun2(lambda x, y: jnp.cos(pi * jnp.cos(pi * x) + pi * jnp.sin(pi * y)), periodic=True)
        f2 = chebfun2(lambda x, y: jnp.cos(pi * jnp.cos(pi * x) + pi * jnp.sin(pi * y)),
                      domain=(-1, 1, -1, 1), periodic=True)
        assert float((f1 - f2).norm()) < 10 * tol                        # pass(9)
        assert isinstance(f1.cols.cols[0].funs[0].tech, Trigtech)        # pass(10)
        assert isinstance(f1.rows.cols[0].funs[0].tech, Trigtech)        # pass(11)

        f = chebfun2(1, coeffs=True)
        assert float((f - 1).norm()) < tol                               # pass(12)
        f = chebfun2("x")
        z = .5 + np.sqrt(3) / 3 * 1j
        assert abs(float(f(z.real, z.imag)) - z.real) < tol              # pass(13)

        m, n = 8, 10
        f = chebfun2(lambda x, y: jnp.cos(x * y), n=(m, n))
        assert _len2(f) == (m, n)                                         # pass(14)
        r = 5
        f = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)), rank=r)
        assert f.rank == r                                               # pass(15)
        r, m, n = 2, 8, 10
        f = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)), rank=r, n=(m, n))
        assert f.rank == r and _len2(f) == (m, n)                        # pass(16)-(17)
        dom = (-1.5, 1.5, -0.5, 0.75)
        r = 3
        f = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)), domain=dom, rank=r)
        assert f.rank == r and tuple(f.domain) == dom                    # pass(18)-(19)
        r, m, n = 3, 20, 37
        f = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)), domain=dom, n=(m, n), rank=r)
        assert f.rank == r and tuple(f.domain) == dom and _len2(f) == (m, n)  # pass(20)
        g = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)))
        r, m, n = 2, 8, 10
        f = chebfun2(g, rank=r, n=(m, n))
        assert f.rank == r and _len2(f) == (m, n)                        # pass(21)
        m, n = 8, 10
        f = chebfun2(lambda x, y: jnp.exp(jnp.cos(x * y)), domain=dom, n=(m, n))
        assert tuple(f.domain) == dom and _len2(f) == (m, n)             # pass(22)-(23)

        with pytest.raises(ValueError, match="constructor:equi"):
            chebfun2(lambda x, y: jnp.sin(x) * jnp.sin(y), equi=True)    # pass(24)

        r = rng.rand(2)
        assert np.linalg.norm(r[:, None] - np.asarray(chebfun2(jnp.asarray(r)).chebpolyval2())) < 10 * tol  # pass(25)
        r = rng.rand(1, 2)
        assert np.linalg.norm(r - np.asarray(chebfun2(jnp.asarray(r)).chebpolyval2())) < 10 * tol  # pass(26)
