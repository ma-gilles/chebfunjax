"""Port of MATLAB Chebfun tests/chebfun/test_fracCalc.m (Fable 5).

Half-derivative of x^n on [0,1] against the Gamma-function closed form
D^q x^n = Gamma(n+1)/Gamma(n+1-q) x^(n-q).

Provenance
----------
MATLAB source : tests/chebfun/test_fracCalc.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gamma

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
Q = float(np.sqrt(2) / 2)
XX = jnp.asarray(np.linspace(0.1, 0.9, 10))


class TestChebfunFracCalc:
    # FIXED in the Fable 5 audit: fracInt now uses Gauss-Jacobi
    # quadrature that absorbs the (x-t)^(mu-1) endpoint singularity
    # into the weight (was Gauss-Legendre on the singular kernel:
    # ~4-digit fracInt, constant-biased fracDiff).
    @pytest.mark.parametrize("n", [4])
    def test_fractional_derivative_of_monomial(self, n):
        x = cj.chebfun(lambda t: t ** n, domain=(0.0, 1.0))
        U = x.fracDiff(Q)
        exact = gamma(n + 1) / gamma(n + 1 - Q) * np.asarray(XX) ** (n - Q)
        err = np.abs(np.asarray(U(XX)) - exact)
        assert float(np.max(err)) < 1e3 * 100 * EPS

    @pytest.mark.xfail(
        reason="the output x^(1-q) has an endpoint singularity; the "
        "smooth-chebfun representation of fracInt's result limits "
        "fracDiff(x, q) to ~2e-8 (needs Singfun-wired factory -- "
        "feature gap). MATLAB stores the result as a singfun.")
    def test_fractional_derivative_of_x(self):
        x = cj.chebfun(lambda t: t, domain=(0.0, 1.0))
        U = x.fracDiff(Q)
        exact = gamma(2) / gamma(2 - Q) * np.asarray(XX) ** (1 - Q)
        err = np.abs(np.asarray(U(XX)) - exact)
        assert float(np.max(err)) < 1e3 * 100 * EPS

    @pytest.mark.parametrize("n", [1, 4])
    def test_fractional_integral_of_monomial(self, n):
        x = cj.chebfun(lambda t: t ** n, domain=(0.0, 1.0))
        U = x.fracInt(Q)
        exact = gamma(n + 1) / gamma(n + 1 + Q) * np.asarray(XX) ** (n + Q)
        err = np.abs(np.asarray(U(XX)) - exact)
        assert float(np.max(err)) < 1e5 * EPS
