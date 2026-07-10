"""Port of MATLAB Chebfun tests/unbndfun/test_diff.m (Opus 4.8).

Self-validating: derivatives are checked against the analytic exact at the
SAME tolerances MATLAB uses.  The later ``pass(k)`` in test_diff.m are in fact
plain ``feval`` checks (they construct ``f`` and evaluate it, never calling
diff); those are ported faithfully as evaluation checks.

Provenance
----------
MATLAB source : tests/unbndfun/test_diff.m
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


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _pts(dc):
    return jnp.asarray(np.linspace(dc[0], dc[1], 100))


class TestUnbndfunDiff:
    # --- Functions on [-inf inf] --------------------------------------
    def test_second_derivative_gaussian(self):
        f = _U(lambda x: jnp.exp(-x ** 2), (-INF, INF))
        g = f.diff(2)
        x = _pts((-1e2, 1e2))
        gexact = 4 * x ** 2 * jnp.exp(-x ** 2) - 2 * jnp.exp(-x ** 2)
        assert _ninf(g(x) - gexact) < 1e3 * EPS * g.vscale

    def test_first_derivative_x2_gaussian(self):
        f = _U(lambda x: x ** 2 * jnp.exp(-x ** 2), (-INF, INF))
        g = f.diff()
        x = _pts((-1e2, 1e2))
        gexact = 2 * x * jnp.exp(-x ** 2) - 2 * x ** 3 * jnp.exp(-x ** 2)
        assert _ninf(g(x) - gexact) < 2e1 * EPS * g.vscale

    def test_first_derivative_odd_decaying(self):
        f = _U(lambda x: (1 - jnp.exp(-x ** 2)) / x, (-INF, INF))
        g = f.diff()
        x = _pts((-1e2, 1e2))
        gexact = 2 * jnp.exp(-x ** 2) + (jnp.exp(-x ** 2) - 1) / x ** 2
        assert _ninf(g(x) - gexact) < 1e2 * EPS * g.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [2 2] "
        "(x^2*(1-exp(-x^2))) grows like x^2 at +-inf and cannot be "
        "represented, so its derivative is unavailable."
    )
    def test_first_derivative_blowup(self):
        f = _U(lambda x: x ** 2 * (1 - jnp.exp(-x ** 2)), (-INF, INF))
        g = f.diff()
        x = _pts((-1e2, 1e2))
        gexact = 2 * x ** 3 * jnp.exp(-x ** 2) - 2 * x * (jnp.exp(-x ** 2) - 1)
        assert _ninf(g(x) - gexact) < 1e7 * EPS * f.vscale

    # --- Functions on [a inf]: pure feval checks (MATLAB pass 5-9) -----
    def test_feval_exp_neg_right_inf(self):
        op = lambda x: jnp.exp(-x)
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_feval_x_exp_neg_right_inf(self):
        op = lambda x: x * jnp.exp(-x)
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 10 * EPS * f.vscale

    def test_feval_odd_decaying_right_inf(self):
        op = lambda x: (1 - jnp.exp(-x)) / x
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_feval_reciprocal_right_inf(self):
        op = lambda x: 1.0 / x
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [0 1] "
        "(x*(5+exp(-x^3)) grows linearly at +inf)."
    )
    def test_feval_blowup_right_inf(self):
        op = lambda x: x * (5 + jnp.exp(-x ** 3))
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e2 * EPS * f.vscale

    # --- Functions on [-inf b]: pure feval checks (MATLAB pass 10-14) --
    def test_feval_exp_left_inf(self):
        op = lambda x: jnp.exp(x)
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < EPS * f.vscale

    def test_feval_x_exp_left_inf(self):
        op = lambda x: x * jnp.exp(x)
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < EPS * f.vscale

    def test_feval_odd_decaying_left_inf(self):
        op = lambda x: (1 - jnp.exp(x)) / x
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < EPS * f.vscale

    def test_feval_reciprocal_left_inf(self):
        op = lambda x: 1.0 / x
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [0 -1] "
        "with a pole at the finite endpoint."
    )
    def test_feval_blowup_left_inf(self):
        b = -3 * np.pi
        op = lambda x: x * (5 + jnp.exp(x ** 3)) / (b - x)
        f = _U(op, (-INF, b))
        x = _pts((-1e6, b))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale
