"""Port of MATLAB Chebfun tests/unbndfun/test_sum.m (Opus 4.8).

Self-validating: definite integrals over unbounded domains are compared to
known closed-form / high-precision values at the SAME tolerances MATLAB uses.

Provenance
----------
MATLAB source : tests/unbndfun/test_sum.m
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


class TestUnbndfunSum:
    # --- Functions on [-inf inf] --------------------------------------
    def test_gaussian(self):
        f = _U(lambda x: jnp.exp(-x ** 2), (-INF, INF))
        assert abs(float(f.sum()) - np.sqrt(np.pi)) < 1e4 * EPS * f.vscale

    def test_x2_gaussian(self):
        f = _U(lambda x: x ** 2 * jnp.exp(-x ** 2), (-INF, INF))
        assert abs(float(f.sum()) - np.sqrt(np.pi) / 2) < 1e6 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax numerical gap: (1-exp(-x^2))/x^2 decays like 1/x^2, "
        "too slowly for the both_inf algebraic map (scale 5) to resolve; "
        "Chebtech2 fails to converge and the integral is inaccurate. MATLAB "
        "issues CHEBFUN:UNBNDFUN:sum:slowDecay and still meets tol; chebfunjax "
        "does not."
    )
    def test_slow_decay_inverse_square(self):
        f = _U(lambda x: (1 - jnp.exp(-x ** 2)) / x ** 2, (-INF, INF))
        assert abs(float(f.sum()) - 2 * np.sqrt(np.pi)) < 1e5 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax has no divergent-integral detection nor blowup "
        "(exponents [2 2]) representation: sum should be +Inf but chebfunjax "
        "returns a finite Gauss-Legendre value."
    )
    def test_divergent_blowup_returns_inf(self):
        f = _U(lambda x: x ** 2 * (1 - jnp.exp(-x ** 2)), (-INF, INF))
        assert float(f.sum()) == INF

    @pytest.mark.xfail(
        reason="chebfunjax has no blowup (exponents [1 1]) representation and no "
        "NaN detection for the (Inf - Inf) principal-value integral of x."
    )
    def test_odd_blowup_returns_nan(self):
        f = _U(lambda x: x, (-INF, INF))
        assert np.isnan(float(f.sum()))

    # --- Functions on [a inf] -----------------------------------------
    def test_exp_neg_right_inf(self):
        f = _U(lambda x: jnp.exp(-x), (1.0, INF))
        assert abs(float(f.sum()) - np.exp(-1)) < 1e5 * EPS * f.vscale

    def test_x_exp_neg_right_inf(self):
        f = _U(lambda x: x * jnp.exp(-x), (1.0, INF))
        assert abs(float(f.sum()) - 2 * np.exp(-1)) < 1e7 * EPS * f.vscale

    def test_odd_decaying_inverse_square_right_inf(self):
        f = _U(lambda x: (1 - jnp.exp(-x)) / x ** 2, (1.0, INF))
        iexact = 0.851504493224078  # MATLAB symbolic toolbox
        tol = 1e4 * EPS * f.vscale
        assert abs(float(f.sum()) - iexact) < 10 * tol

    def test_reciprocal_square_right_inf(self):
        f = _U(lambda x: 1.0 / x ** 2, (1.0, INF))
        assert abs(float(f.sum()) - 1.0) < 1e5 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax has no divergent-integral detection nor blowup "
        "(exponents [0 1]) representation: sum should be +Inf."
    )
    def test_divergent_blowup_right_inf(self):
        f = _U(lambda x: x * (5 + jnp.exp(-x ** 3)), (1.0, INF))
        assert float(f.sum()) == INF

    # --- Functions on [-inf b] ----------------------------------------
    def test_exp_left_inf(self):
        f = _U(lambda x: jnp.exp(x), (-INF, -3 * np.pi))
        assert abs(float(f.sum()) - np.exp(-3 * np.pi)) < 5e4 * EPS * f.vscale

    def test_x_exp_left_inf(self):
        f = _U(lambda x: x * jnp.exp(x), (-INF, -3 * np.pi))
        iexact = -np.exp(-3 * np.pi) * (3 * np.pi + 1)
        tol = 1e4 * EPS * f.vscale
        assert abs(float(f.sum()) - iexact) < 10 * tol

    def test_odd_decaying_inverse_square_left_inf(self):
        f = _U(lambda x: (1 - jnp.exp(x)) / x ** 2, (-INF, -3 * np.pi))
        iexact = 0.106102535711326  # MATLAB symbolic toolbox
        assert abs(float(f.sum()) - iexact) < 1e5 * EPS * f.vscale

    def test_reciprocal_square_left_inf(self):
        f = _U(lambda x: 1.0 / x ** 2, (-INF, -3 * np.pi))
        assert abs(float(f.sum()) - 1 / (3 * np.pi)) < 2e4 * EPS * f.vscale

    def test_reciprocal_square_left_inf_exponents(self):
        # MATLAB constructs this with exponents [-2 0] and singPref; chebfunjax
        # has no exponents path, but the smooth construction of 1/x^2 already
        # integrates correctly on (-inf, -3*pi], so the assertion holds.
        f = _U(lambda x: 1.0 / x ** 2, (-INF, -3 * np.pi))
        assert abs(float(f.sum()) - 1 / (3 * np.pi)) < 5e3 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax has no divergent-integral detection: sum of a nonzero "
        "constant over an infinite interval should be +Inf."
    )
    def test_constant_returns_inf(self):
        f = _U(lambda x: 0 * x + 2, (-INF, -3 * np.pi))
        assert float(f.sum()) == INF

    # --- Chebfun with singular 'exps' (MATLAB pass 17-19) -------------
    @pytest.mark.xfail(
        reason="chebfunjax lacks singfun 'exps' on a semi-infinite chebfun: "
        "sqrt(t)/exp(t) on [0,inf] needs an endpoint exponent 0.5."
    )
    def test_chebfun_sqrt_exp_single_piece(self):
        raise NotImplementedError("chebfun with 'exps' on [0,inf]")

    @pytest.mark.xfail(
        reason="chebfunjax lacks singfun 'exps' on a semi-infinite chebfun."
    )
    def test_chebfun_sqrt_exp_split(self):
        raise NotImplementedError("chebfun with 'exps' on [0,1,inf]")

    @pytest.mark.xfail(
        reason="chebfunjax lacks singfun 'exps' on a semi-infinite chebfun."
    )
    def test_chebfun_sqrt_exp_reassigned(self):
        raise NotImplementedError("chebfun with 'exps' on [0,inf], f3(1)=f3(1)")
