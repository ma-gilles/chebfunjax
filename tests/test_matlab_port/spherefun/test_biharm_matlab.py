"""Port of MATLAB Chebfun tests/spherefun/test_biharm.m.

Provenance
----------
MATLAB source : tests/spherefun/test_biharm.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.utils.quadrature import trigpts


def _source_grid():
    lam = trigpts(12, [-jnp.pi, jnp.pi])[0]
    th = jnp.linspace(0.0, jnp.pi, 6)
    return lam, th


def _source_sample_error(expected, approximation, lam, th):
    L, T = np.meshgrid(np.asarray(lam), np.asarray(th))
    expected_values = np.asarray(expected(jnp.asarray(L), jnp.asarray(T)))
    actual = np.asarray(approximation.fevalm(lam, th))
    return float(np.linalg.norm((expected_values-actual).ravel(), ord=np.inf))


class TestSpherefunBiharm:
    @pytest.mark.parametrize("ell", [1, 2, 4, 5],
                             ids=lambda ell: f"outer_ell_{ell}")
    def test_original_harmonic_source_predicates(self, ell):
        tol = 1e4 * ChebfunPref().cheb2Prefs.chebfun2eps
        lam, th = _source_grid()
        for m in range(ell+1):
            f = Spherefun.sphharm(ell, m)
            bih = f.biharm()
            # Source pass column 1: pivot-value cardinality is preserved.
            assert np.asarray(bih.pivot_values).size == np.asarray(f.pivot_values).size, (
                f"ell={ell}, m={m}: pivotValues count differs")
            scale = (ell*(ell+1))**2
            # Match MATLAB's pre-evaluation Spherefun scaling: scale*f.
            expected = scale*f
            error = _source_sample_error(expected, bih, lam, th)
            # Source pass column 2: SampleError < (ell*(ell+1))^2*tol.
            assert error < scale*tol, (
                f"ell={ell}, m={m}: SampleError={error}, bound={scale*tol}")
