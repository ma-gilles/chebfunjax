"""Port of MATLAB Chebfun tests/spherefunv/test_vectorRelations.m (Fable 5).

FIXED: with the 3-Cartesian-component Spherefunv (grad/gradient, div, curl,
vort) and Spherefun.curl, the standard surface vector-calculus identities
hold to spectral accuracy.

Provenance
----------
MATLAB source : tests/spherefunv/test_vectorRelations.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun

from ._helpers import EPS, snorm

TOL = 3e3 * EPS


class TestSpherefunvVectorRelations:
    def _f(self) -> Spherefun:
        return Spherefun.from_function(
            lambda lam, th: jnp.cos((jnp.cos(lam) * jnp.sin(th) + 0.1)
                                    * (jnp.sin(lam) * jnp.sin(th))
                                    * jnp.cos(th)))

    def test_div_grad_is_laplacian(self):
        # pass(1): div(grad(f)) == laplacian(f).
        f = self._f()
        assert snorm(f.gradient().div() - f.laplacian()) < TOL

    def test_div_curl_is_zero(self):
        # pass(2): div(curl(f)) == 0 for a scalar field f.
        f = self._f()
        assert snorm(f.curl().div()) < TOL

    def test_vort_grad_is_zero(self):
        # pass(3): vort(grad(f)) == 0.
        f = self._f()
        assert snorm(f.gradient().vort()) < TOL
