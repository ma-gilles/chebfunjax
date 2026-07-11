"""Port of MATLAB Chebfun tests/chebfun/test_chebcoeffs.m (Fable 5).

chebfunjax exposes coefficients via .coeffs (single-piece); MATLAB's
Bessel closed forms for the Chebyshev coefficients of cos(x) are the
reference.

Provenance
----------
MATLAB source : tests/chebfun/test_chebcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunChebcoeffs:
    def test_cos_coefficients_bessel_closed_form(self):
        f = cj.chebfun(jnp.cos)
        c = np.asarray(f.coeffs())[:5] if callable(
            getattr(f, "coeffs", None)) else np.asarray(f.coeffs)[:5]
        c_exact = np.array([jv(0, 1), 0.0, -2 * jv(2, 1), 0.0,
                            2 * jv(4, 1)])
        assert float(np.max(np.abs(c - c_exact))) < 100 * EPS

    def test_truncation_argument(self):
        pytest.skip("chebfunjax coeffs has no n-truncation argument "
                    "(slice instead)")
