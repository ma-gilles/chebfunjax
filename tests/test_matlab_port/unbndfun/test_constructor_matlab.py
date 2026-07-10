"""Port of MATLAB Chebfun tests/unbndfun/test_constructor.m (Opus 4.8).

Self-validating: the constructor is checked by evaluating the resulting
Unbndfun at deterministic points across the (truncated) unbounded domain and
comparing to the analytic operator at the SAME tolerance MATLAB uses
(multiples of ``eps*vscale``).  Test points are our own linspace over the
finite check-window ``domCheck`` (the error bound holds at any point).

Provenance
----------
MATLAB source : tests/unbndfun/test_constructor.m
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


class TestUnbndfunConstructor:
    # --- Functions on [-inf inf] --------------------------------------
    def test_gaussian_both_inf(self):
        op = lambda x: jnp.exp(-x ** 2)
        f = _U(op, (-INF, INF))
        x = _pts((-1e2, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_x2_gaussian_both_inf(self):
        op = lambda x: x ** 2 * jnp.exp(-x ** 2)
        f = _U(op, (-INF, INF))
        x = _pts((-1e2, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_odd_decaying_both_inf(self):
        op = lambda x: (1 - jnp.exp(-x ** 2)) / x
        f = _U(op, (-INF, INF))
        x = _pts((-1e2, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [2 2] with "
        "pref.blowup (x^2*(1-exp(-x^2)) grows like x^2 at +-inf) is not "
        "representable; Unbndfun.from_function has no 'exponents' path."
    )
    def test_blowup_both_inf(self):
        op = lambda x: x ** 2 * (1 - jnp.exp(-x ** 2))
        f = _U(op, (-INF, INF))
        x = _pts((-1e2, 1e2))
        assert _ninf(f(x) - op(x)) < 1e5 * EPS * f.vscale

    # --- Functions on [a inf] -----------------------------------------
    def test_exp_neg_right_inf(self):
        op = lambda x: jnp.exp(-x)
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_x_exp_neg_right_inf(self):
        op = lambda x: x * jnp.exp(-x)
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_odd_decaying_right_inf(self):
        op = lambda x: (1 - jnp.exp(-x)) / x
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_reciprocal_right_inf(self):
        op = lambda x: 1.0 / x
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [0 1] with "
        "x*(5+exp(-x^3)) grows linearly at +inf; no 'exponents' path."
    )
    def test_blowup_right_inf(self):
        op = lambda x: x * (5 + jnp.exp(-x ** 3))
        f = _U(op, (1.0, INF))
        x = _pts((1, 1e2))
        assert _ninf(f(x) - op(x)) < 1e2 * EPS * f.vscale

    # --- Functions on [-inf b] ----------------------------------------
    def test_exp_left_inf(self):
        op = lambda x: jnp.exp(x)
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_x_exp_left_inf(self):
        op = lambda x: x * jnp.exp(x)
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_odd_decaying_left_inf(self):
        op = lambda x: (1 - jnp.exp(x)) / x
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    def test_reciprocal_left_inf(self):
        op = lambda x: 1.0 / x
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular/blowup Unbndfun: exponents [0 -1] with "
        "a pole at the finite endpoint is not representable."
    )
    def test_blowup_left_inf(self):
        b = -3 * np.pi
        op = lambda x: x * (5 + jnp.exp(x ** 3)) / (b - x)
        f = _U(op, (-INF, b))
        x = _pts((-1e6, b))
        assert _ninf(f(x) - op(x)) < 1e1 * EPS * f.vscale

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Unbndfun: [exp(x) x*exp(x) "
        "(1-exp(x))/x] is a 3-column fun; Unbndfun wraps a single scalar "
        "Chebtech2."
    )
    def test_array_valued_left_inf(self):
        op = lambda x: jnp.stack(
            [jnp.exp(x), x * jnp.exp(x), (1 - jnp.exp(x)) / x], axis=-1
        )
        f = _U(op, (-INF, -3 * np.pi))
        x = _pts((-1e6, -3 * np.pi))
        assert _ninf(np.asarray(f(x)) - np.asarray(op(x))) < 1e2 * EPS * f.vscale

    # --- MISC: bounded-domain error -----------------------------------
    def test_bounded_domain_raises(self):
        # MATLAB: CHEBFUN:UNBNDFUN:unbndfun:boundedDomain.
        # chebfunjax raises ValueError (domain has no infinite endpoint).
        with pytest.raises(ValueError):
            _U(lambda x: jnp.exp(-x ** 2), (0.0, 1.0))
