"""Port of MATLAB Chebfun tests/unbndfun/test_cumsum.m (Opus 4.8).

Self-validating: the indefinite integral is checked against the analytic
antiderivative at the SAME tolerances MATLAB uses.  Note that MATLAB's exact
antiderivatives are pinned so that ``F`` matches ``cumsum`` up to the constant
chosen by the algorithm (they encode ``F(left endpoint) = 0``).

Provenance
----------
MATLAB source : tests/unbndfun/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from jax.scipy.special import erf

from chebfunjax.domain import Domain
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
INF = np.inf


def _U(op, dom):
    return Unbndfun.from_function(op, Domain(dom))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _pts(dc):
    return jnp.asarray(np.linspace(dc[0], dc[1], 100))


class TestUnbndfunCumsum:
    # --- Functions on [-inf inf] --------------------------------------
    def test_cumsum_gaussian(self):
        # cumsum(exp(-x^2)) = sqrt(pi)*erf(x)/2 + sqrt(pi)/2 (with F(-inf)=0)
        f = _U(lambda x: jnp.exp(-x ** 2), (-INF, INF))
        g = f.cumsum()
        x = _pts((-1e2, 1e2))
        gexact = np.sqrt(np.pi) * erf(x) / 2 + np.sqrt(np.pi) / 2
        assert _ninf(g(x) - gexact) < 5e4 * EPS * g.vscale

    def test_placeholder_blowup_cumsum(self):
        # MATLAB pass(2) = 1 is a hard-coded pass: the blowup cumsum test is
        # commented out there ("[TODO]: Revive when log is ready").  We keep a
        # faithful always-passing placeholder for the same assertion index.
        assert True

    # --- Functions on [a inf] -----------------------------------------
    def test_cumsum_x_exp_neg(self):
        # cumsum(x*exp(-x)) = -exp(-x)*(x+1) + 2*exp(-1) (with F(1)=0)
        f = _U(lambda x: x * jnp.exp(-x), (1.0, INF))
        g = f.cumsum()
        x = _pts((1, 1e2))
        gexact = -jnp.exp(-x) * (x + 1) + 2 * np.exp(-1)
        assert _ninf(g(x) - gexact) < 1e6 * EPS * g.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [0 1] on "
        "5*x (linear growth) plus reliance on get(g,'lval'); not "
        "representable."
    )
    def test_cumsum_blowup_right_inf(self):
        f = _U(lambda x: 5 * x, (1.0, INF))
        g = f.cumsum()
        x = _pts((1, 1e2))
        # opg = 5*x^2/2 - 5/2 + lval  (lval from the algorithm)
        gexact = 5 * x ** 2 / 2 - 5 / 2
        assert _ninf(g(x) - gexact) < 200 * EPS * g.vscale

    # --- Functions on [-inf b] ----------------------------------------
    def test_cumsum_exp_left_inf(self):
        # cumsum(exp(x)) = exp(x) (with F(-inf)=0)
        f = _U(lambda x: jnp.exp(x), (-INF, -3 * np.pi))
        g = f.cumsum()
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(g(x) - jnp.exp(x)) < 1e5 * EPS * g.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun: [exp(x) x*exp(x)] is a "
        "2-column fun."
    )
    def test_cumsum_array_valued(self):
        op = lambda x: jnp.stack([jnp.exp(x), x * jnp.exp(x)], axis=-1)
        f = _U(op, (-INF, -3 * np.pi))
        g = f.cumsum()
        x = _pts((-1e6, -3 * np.pi))
        gexact = jnp.stack([jnp.exp(x), jnp.exp(x) * (x - 1)], axis=-1)
        assert _ninf(np.asarray(g(x)) - np.asarray(gexact)) < 5e5 * EPS * g.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun and a dim=2 cumsum "
        "(cumulative sum across columns)."
    )
    def test_cumsum_over_columns(self):
        # MATLAB: f = unbndfun(@(x) [exp(x) x.*exp(x)], [-inf -3*pi]);
        #         h = cumsum(f, 2) -- cumulative sum across the 2 columns.
        raise NotImplementedError("array-valued Unbndfun + dim=2 cumsum")
