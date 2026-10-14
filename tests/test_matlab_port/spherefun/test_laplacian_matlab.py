"""Port of MATLAB Chebfun tests/spherefun/test_laplacian.m.

Provenance
----------
MATLAB source : tests/spherefun/test_laplacian.m
Chebfun commit: 7574c77

Each outer degree is a separate pytest case so qualification can run each
case in a fresh bounded child and release JAX caches serially.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils.quadrature import trigpts


def _source_grid():
    # Native getPoints(6,6): trigpts(2*n,[-pi,pi]) and linspace(0,pi,m).
    lam = trigpts(12, [-jnp.pi, jnp.pi])[0]
    th = jnp.linspace(0.0, jnp.pi, 6)
    return lam, th


def _source_sample_error(expected, approximation, lam, th):
    L, T = np.meshgrid(np.asarray(lam), np.asarray(th))
    expected_values = np.asarray(expected(jnp.asarray(L), jnp.asarray(T)))
    actual = np.asarray(approximation.fevalm(lam, th))
    return float(np.linalg.norm((expected_values-actual).ravel(), ord=np.inf))


class TestSpherefunLaplacian:
    @pytest.mark.parametrize("ell", [1, 2, 4, 5, 7, 8, 9],
                             ids=lambda ell: f"outer_ell_{ell}")
    def test_original_harmonic_source_predicates(self, ell):
        tol = 1e4 * ChebfunPref().cheb2Prefs.chebfun2eps
        lam, th = _source_grid()
        for m in range(-ell, ell+1):
            f = Spherefun.sphharm(ell, m)
            lap = f.laplacian()
            # Source pass column 1: numel(lap.pivotValues) == numel(f.pivotValues).
            assert np.asarray(lap.pivot_values).size == np.asarray(f.pivot_values).size, (
                f"ell={ell}, m={m}: pivotValues count differs")
            scale = ell*(ell+1)
            # Match MATLAB's pre-evaluation Spherefun scaling: -scale*f.
            expected = -scale*f
            error = _source_sample_error(expected, lap, lam, th)
            # Source pass column 2: SampleError < ell*(ell+1)*tol.
            assert error < scale*tol, (
                f"ell={ell}, m={m}: SampleError={error}, bound={scale*tol}")

    def test_original_gaussian_source_predicate(self):
        tol = 1e4 * ChebfunPref().cheb2Prefs.chebfun2eps
        lam, th = _source_grid()
        lam0 = jnp.pi/jnp.sqrt(2.0)
        th0 = (jnp.sqrt(5.0)-2.0)/2.0
        sig2 = 1.0

        def r2(L,T):
            return 2*(1-(jnp.sin(T)*jnp.sin(th0))*jnp.cos(L-lam0)
                      -jnp.cos(T)*jnp.cos(th0))

        f = Spherefun.from_function(lambda L,T: jnp.exp(-sig2*r2(L,T)))

        def exact(L,T):
            R = r2(L,T)
            return sig2*jnp.exp(-sig2*R)*(-4+R*(2-sig2*(-4+R)))

        lap = f.laplacian()
        error = _source_sample_error(exact, lap, lam, th)
        assert error < 10*tol, f"Gaussian SampleError={error}, bound={10*tol}"

    @pytest.mark.parametrize("l,m", [(1, 0), (2, 1), (3, 3), (5, -2)])
    def test_supplemental_previous_pointwise_identity(self, l, m):
        # Preserve prior non-source (3,3) check as a labeled supplement.
        Y = Spherefun.sphharm(l, m)
        lam, th = jnp.asarray(0.5), jnp.asarray(1.2)
        got = float(Y.laplacian()(lam, th))
        want = -l*(l+1)*float(Y(lam, th))
        assert abs(got-want) < 1e-8*max(abs(want), 1.0)
