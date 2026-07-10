"""Port of MATLAB Chebfun tests/unbndfun/test_innerProduct.m (Opus 4.8).

Self-validating: L2 inner products over unbounded domains are compared to
known closed forms at the SAME tolerances MATLAB uses.

Provenance
----------
MATLAB source : tests/unbndfun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf


def _U(op, dom):
    return Unbndfun.from_function(op, Domain(dom))


class TestUnbndfunInnerProduct:
    def test_both_inf(self):
        f = _U(lambda x: 2 - jnp.exp(-x ** 2), (-INF, INF))
        g = _U(lambda x: jnp.exp(-x ** 2), (-INF, INF))
        iexact = (np.sqrt(np.pi) * (4 - np.sqrt(2))) / 2
        tol = 2e7 * max(EPS * f.vscale, EPS * g.vscale)
        assert abs(float(f.inner(g)) - iexact) < tol

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: f = x on [1,inf) with "
        "exponents [0 1] (linear growth) is not representable."
    )
    def test_right_inf_blowup_factor(self):
        f = _U(lambda x: x, (1.0, INF))
        g = _U(lambda x: jnp.exp(-x), (1.0, INF))
        iexact = 2 * np.exp(-1)
        tol = 2e8 * max(EPS * f.vscale, EPS * g.vscale)
        assert abs(float(f.inner(g)) - iexact) < tol

    @pytest.mark.xfail(
        reason="chebfunjax numerical gap: the integrand (1/x)(2/x)*dx/dy for the "
        "inner product decays like 1/x^2 and does not vanish at the reference "
        "endpoint, where 0*inf sanitisation to 0 corrupts the fixed-length "
        "Chebtech2; the integral is off by ~1e-4 vs MATLAB's tol."
    )
    def test_left_inf_reciprocal(self):
        f = _U(lambda x: 1.0 / x, (-INF, -3 * np.pi))
        g = _U(lambda x: 2.0 / x, (-INF, -3 * np.pi))
        iexact = 2 / (3 * np.pi)
        assert abs(float(f.inner(g)) - iexact) < 1e5 * EPS * f.vscale
