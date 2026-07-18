"""Port of MATLAB Chebfun tests/chebfun/test_chebpoly.m (Fable 5).

MATLAB ``chebpoly(f, n)`` is a (deprecated) accessor for the first ``n``
Chebyshev coefficients of ``f`` (returned highest-degree-first via ``rev``).
chebfunjax exposes the ascending coefficients via the ``.coeffs`` property; the
scalar case is already covered by test_chebcoeffs, and array-valued CHEBFUN now
gives an ``(n, m)`` coefficient matrix, so the 1st-kind smooth cases (MATLAB
pass 1 and 2) are ported here directly (FIXED, Fable 5, Big-Three array-valued
epic).

Remaining skips (precise gaps, not array-valuedness):
- 'kind', 2 variants (pass 3, 4, 7, 8): chebfunjax has no 2nd-kind Chebyshev
  coefficient accessor.
- piecewise ``abs(x)`` on ``[-1 0 1]`` (pass 5, 6): ``.coeffs`` is single-piece
  only; there is no global-expansion coefficient accessor for a multi-piece
  chebfun.

Provenance
----------
MATLAB source : tests/chebfun/test_chebpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)

# First 5 first-kind Chebyshev coefficients of cos(x) (ascending, c_0 first).
# MATLAB's c_exact = [J0(1); 0; -2 J2(1); 0; 2(-23 J0(1) + 40 J1(1))].
_C_EXACT = np.array(
    [jv(0, 1), 0.0, -2 * jv(2, 1), 0.0, 2 * (-23 * jv(0, 1) + 40 * jv(1, 1))]
)


class TestChebfunChebpoly:
    def test_scalar_1st_kind(self):
        # pass(1): first 5 first-kind Chebyshev coeffs of cos(x).
        f = cj.chebfun(np.cos)
        c = np.asarray(f.coeffs)[:5]
        assert float(np.max(np.abs(c - _C_EXACT))) < 1e2 * f.vscale * EPS

    def test_array_valued_1st_kind(self):
        # pass(2): array-valued [cos cos] -> coeffs match per column.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        import jax.numpy as jnp

        g = cj.chebfun(lambda x: jnp.stack([jnp.cos(x), jnp.cos(x)], axis=-1))
        c = np.asarray(g.coeffs)[:5]
        assert c.shape == (5, 2)
        assert float(np.max(np.abs(c - _C_EXACT[:, None]))) < 1e2 * g.vscale * EPS

    def test_kind_2(self):
        # pass(3, 4, 7, 8): chebpoly(..., 'kind', 2).
        pytest.skip(
            "chebfunjax has no 2nd-kind Chebyshev coefficient accessor "
            "(only 1st-kind .coeffs)"
        )

    def test_piecewise(self):
        # pass(5, 6): chebpoly of piecewise abs(x) on [-1 0 1].
        pytest.skip(
            "chebfunjax .coeffs is single-piece; no global-expansion coefficient "
            "accessor for a multi-piece chebfun"
        )
