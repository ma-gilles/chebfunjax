"""Port of MATLAB Chebfun tests/classicfun/test_times.m (Opus 4.8).

Self-validating: pointwise multiplication (fun.*scalar, fun.*fun) is checked
against the analytic product at the SAME tolerances MATLAB uses.  ``isequal``
is reproduced by comparing Chebyshev coefficients + domains.

Provenance
----------
MATLAB source : tests/classicfun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.unbndfun import Unbndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))
ALPHA = -0.194758928283640 + 0.075474485412665j
INF = np.inf


def _bf(op):
    return Bndfun.from_function(op, DOM)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _isequal(f, g):
    if f.domain != g.domain:
        return False
    fc, gc = np.asarray(f.coeffs), np.asarray(g.coeffs)
    return fc.shape == gc.shape and np.array_equal(fc, gc)


def _mult_fun_by_fun(f, f_op, g, g_op):
    h = f * g
    hexact = f_op(X) * g_op(X)
    tol = 10 * h.vscale * EPS
    return _ninf(h(X) - hexact) < 1e4 * tol


class TestClassicfunTimes:
    @pytest.mark.xfail(
        reason="chebfunjax has no empty-fun construction (bndfun() with no "
        "args)."
    )
    def test_empty(self):
        raise NotImplementedError("empty Bndfun")

    # --- multiply by scalar -------------------------------------------
    def test_mult_scalar_isequal(self):
        f = _bf(jnp.sin)
        assert _isequal(f * ALPHA, ALPHA * f)

    def test_mult_scalar_norm(self):
        f = _bf(jnp.sin)
        g1 = f * ALPHA
        gexact = jnp.sin(X) * ALPHA
        assert _ninf(g1(X) - gexact) < 10 * g1.vscale * EPS

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_mult_scalar_isequal(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_mult_scalar_norm(self):
        raise NotImplementedError

    # --- multiply by a constant function ------------------------------
    def test_mult_by_constant_function(self):
        f_op = lambda x: jnp.sin(x)
        g_op = lambda x: ALPHA * jnp.ones_like(x)
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_by_constant_row(self):
        raise NotImplementedError

    # --- products of two funs -----------------------------------------
    def test_mult_ones_by_ones(self):
        f_op = lambda x: jnp.ones_like(x)
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(f_op), f_op)

    def test_mult_exp_recip(self):
        f_op = lambda x: jnp.exp(x) - 1
        g_op = lambda x: 1.0 / (1 + x ** 2)
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

    def test_mult_exp_cos1e4(self):
        f_op = lambda x: jnp.exp(x) - 1
        g_op = lambda x: jnp.cos(1e4 * x)
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

    def test_mult_exp_sinh_complex(self):
        f_op = lambda x: jnp.exp(x) - 1
        g_op = lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

    # --- array-valued products (pass 12-14) ---------------------------
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_commute(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_product_norm(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_product_norm_2(self):
        raise NotImplementedError

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: a dimension-mismatch "
        "product should raise CHEBFUN:CHEBTECH:times:dim2."
    )
    def test_dimension_mismatch(self):
        raise NotImplementedError

    # --- complex, positivity ------------------------------------------
    def test_mult_sinh_complex_self(self):
        f_op = lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))
        f = _bf(f_op)
        assert _mult_fun_by_fun(f, f_op, f, f_op)

    @pytest.mark.xfail(
        reason="chebfunjax has no conj() method on Bndfun/Chebtech2, so "
        "f.*conj(f) (with the positivity check) cannot be formed."
    )
    def test_mult_by_conj_1(self):
        raise NotImplementedError("conj not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax has no conj() method on Bndfun/Chebtech2."
    )
    def test_mult_by_conj_2(self):
        raise NotImplementedError("conj not implemented")

    def test_mult_expix_minus1_self(self):
        # MATLAB pass(19:20): identical assertion recorded twice.
        f_op = lambda x: jnp.exp(1j * x) - 1
        f = _bf(f_op)
        assert _mult_fun_by_fun(f, f_op, f, f_op)

    # --- multiplication vs direct construction ------------------------
    def test_mult_vs_direct(self):
        f_op = lambda x: jnp.exp(1j * x) - 1
        g_op = lambda x: 1.0 / (1 + x ** 2)
        f = _bf(f_op)
        g = _bf(g_op)
        h1 = f * g
        h2 = _bf(lambda x: (jnp.exp(1j * x) - 1) * (1.0 / (1 + x ** 2)))
        assert _ninf(h1(X) - h2(X)) < 2e1 * EPS * h1.vscale

    # --- happy .* unhappy ---------------------------------------------
    @pytest.mark.xfail(
        reason="chebfunjax arithmetic does not propagate ishappy=False from an "
        "unhappy operand."
    )
    def test_unhappy_result_1(self):
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = _bf(lambda x: jnp.cos(x + 1)) * g
        assert (not g.ishappy) and (not h.ishappy)

    @pytest.mark.xfail(
        reason="chebfunjax arithmetic does not propagate ishappy=False."
    )
    def test_unhappy_result_2(self):
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = g * _bf(lambda x: jnp.cos(x + 1))
        assert (not g.ishappy) and (not h.ishappy)

    # --- singular (pass 24-25) ----------------------------------------
    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (exponents) Bndfun: (x-b)^p * scalar."
    )
    def test_singular_scalar(self):
        raise NotImplementedError("singular Bndfun times scalar")

    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (exponents) Bndfun: product of two "
        "singular funs."
    )
    def test_singular_product(self):
        raise NotImplementedError("singular Bndfun product")

    # --- Unbndfun multiplication (pass 26) ----------------------------
    def test_unbndfun_times(self):
        dom = Domain((-INF, INF))
        f = Unbndfun.from_function(lambda x: x ** 2 * jnp.exp(-x ** 2), dom)
        g = Unbndfun.from_function(lambda x: (1 - jnp.exp(-x ** 2)) / x, dom)
        h = f * g
        x = jnp.asarray(np.linspace(-1e2, 1e2, 100))
        hexact = x * jnp.exp(-x ** 2) * (1 - jnp.exp(-x ** 2))
        assert _ninf(h(x) - hexact) < 2 * EPS * f.vscale
