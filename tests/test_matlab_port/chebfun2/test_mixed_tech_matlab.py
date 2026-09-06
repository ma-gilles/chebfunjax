"""Port of MATLAB Chebfun tests/chebfun2/test_mixed_tech.m (Fable 5).

MATLAB's flags ``'trigx'``/``'periodicy'``/``'equix'``/``'coeffsy'`` ...
are keyword arguments of the same names; ``chebpts2(n, m)`` is the
tensor Chebyshev grid and ``trigpts`` the equispaced trig grid.

Provenance
----------
MATLAB source : tests/chebfun2/test_mixed_tech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts

jax.config.update("jax_enable_x64", True)

pi = np.pi


def _trigpts(n):
    return -1.0 + 2.0 * np.arange(n) / n


def _grids(n, m):
    xc, yc = np.meshgrid(np.asarray(chebpts(n, kind=2)), np.asarray(chebpts(m, kind=2)))
    xt, yt = np.meshgrid(_trigpts(n), _trigpts(m))
    xe, ye = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, m))
    return xc, yc, xt, yt, xe, ye


def _v2c(V):
    return np.asarray(Chebtech2.vals2coeffs(jnp.asarray(V)))


class TestChebfun2MixedTech:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        u = lambda x, y: jnp.sin(2 * pi * x) * jnp.cos(2 * pi * y)  # noqa: E731
        f1 = chebfun2(u, trigx=True)
        f2 = chebfun2(u, periodicy=True)
        f3 = chebfun2(u, trig=True)
        f4 = chebfun2(u)
        assert float((f1 - f2).norm()) < tol
        assert float((f2 - f3).norm()) < tol
        assert float((f3 - f4).norm()) < tol

        m, n = 31, 32
        xc, yc, xt, yt, xe, ye = _grids(n, m)
        U = lambda X, Y: np.asarray(u(jnp.asarray(X), jnp.asarray(Y)))  # noqa: E731
        f1 = chebfun2(U(xt, yc), trigx=True)
        f2 = chebfun2(U(xc, yt), periodicy=True)
        f3 = chebfun2(U(xt, yt), trig=True)
        f4 = chebfun2(U(xc, yc))
        assert float((f1 - f2).norm()) < tol
        assert float((f2 - f3).norm()) < tol
        assert float((f3 - f4).norm()) < tol

        u = lambda x, y: jnp.sin(x * y)  # noqa: E731
        U = lambda X, Y: np.asarray(u(jnp.asarray(X), jnp.asarray(Y)))  # noqa: E731
        f1 = chebfun2(U(xe, yc), equix=True)
        f2 = chebfun2(U(xc, ye), equiy=True)
        f3 = chebfun2(U(xe, ye), equi=True)
        f4 = chebfun2(U(xc, yc))
        assert float((f1 - f2).norm()) < tol
        assert float((f2 - f3).norm()) < tol
        assert float((f3 - f4).norm()) < tol

        vals_vals = U(xc, yc)
        vals_coeffs = _v2c(vals_vals)
        coeffs_vals = _v2c(vals_vals.T).T
        coeffs_coeffs = _v2c(vals_coeffs.T).T
        f1 = chebfun2(u)
        f2 = chebfun2(coeffs_vals, coeffsx=True)
        f3 = chebfun2(vals_coeffs, coeffsy=True)
        f4 = chebfun2(coeffs_coeffs, coeffs=True)
        assert float((f1 - f2).norm()) < tol
        assert float((f2 - f3).norm()) < tol
        assert float((f3 - f4).norm()) < tol
