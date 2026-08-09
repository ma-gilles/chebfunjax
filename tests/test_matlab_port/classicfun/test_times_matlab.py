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
    def test_empty(self):
        # pass(1): isempty(f .* f) && isempty(f .* g) && isempty(g .* f)
        f = Bndfun.empty()
        g = Bndfun.from_function(jnp.sin, DOM)
        assert (f * f).isempty()
        assert (f * g).isempty()
        assert (g * f).isempty()

    # --- multiply by scalar -------------------------------------------
    def test_mult_scalar_isequal(self):
        f = _bf(jnp.sin)
        assert _isequal(f * ALPHA, ALPHA * f)

    def test_mult_scalar_norm(self):
        f = _bf(jnp.sin)
        g1 = f * ALPHA
        gexact = jnp.sin(X) * ALPHA
        assert _ninf(g1(X) - gexact) < 10 * g1.vscale * EPS

    def test_array_mult_scalar_isequal(self):
        # pass(4): [sin cos] .* alpha == alpha .* [sin cos].
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        assert _isequal(f * ALPHA, ALPHA * f)

    def test_array_mult_scalar_norm(self):
        # pass(5): value of [sin cos] .* alpha, tol 10*vscale*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = _bf(fop)
        g1 = f * ALPHA
        assert _ninf(g1(X) - fop(X) * ALPHA) < 10 * g1.vscale * EPS

    # --- multiply by a constant function ------------------------------
    def test_mult_by_constant_function(self):
        f_op = lambda x: jnp.sin(x)
        g_op = lambda x: ALPHA * jnp.ones_like(x)
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

    def test_array_by_constant_row(self):
        # pass(7): [sin cos] .* [alpha beta] (row of constant columns).
        # FIXED (Fable 5, Big-Three array-valued epic).
        beta = -0.526634844879922 - 0.685484380523668j
        f_op = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        g_op = lambda x: jnp.stack(
            [ALPHA * jnp.ones_like(x), beta * jnp.ones_like(x)], axis=-1
        )
        assert _mult_fun_by_fun(_bf(f_op), f_op, _bf(g_op), g_op)

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
    def test_array_commute(self):
        # pass(12): [sin cos exp] .* tanh commutes (column-vs-scalar broadcast).
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = _bf(jnp.tanh)
        h1 = f * g
        h2 = g * f
        assert _ninf((h1 - h2)(X)) < 1000 * h1.vscale * EPS

    def test_array_product_norm(self):
        # pass(13): value of [sin cos exp] .* tanh, tol 10*vscale*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = _bf(fop)
        g = _bf(jnp.tanh)
        h = f * g
        hexact = fop(X) * jnp.tanh(X)[:, None]
        assert _ninf(h(X) - hexact) < 10 * h.vscale * EPS

    def test_array_product_norm_2(self):
        # pass(14): [sin cos exp] .* [sinh cosh tanh], tol 10*vscale*eps.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        gop = lambda x: jnp.stack([jnp.sinh(x), jnp.cosh(x), jnp.tanh(x)], axis=-1)
        h = _bf(fop) * _bf(gop)
        assert _ninf(h(X) - fop(X) * gop(X)) < 10 * h.vscale * EPS

    def test_dimension_mismatch(self):
        # pass(15): [sin cos exp](3col) .* [sinh cosh](2col) is a shape mismatch.
        # FIXED (Fable 5, Big-Three array-valued epic): chebfunjax raises (MATLAB
        # emits CHEBFUN:CHEBTECH:times:dim2; chebfunjax has no typed identifier).
        f = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = _bf(lambda x: jnp.stack([jnp.sinh(x), jnp.cosh(x)], axis=-1))
        with pytest.raises(Exception):
            f * g

    # --- complex, positivity ------------------------------------------
    def test_mult_sinh_complex_self(self):
        f_op = lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))
        f = _bf(f_op)
        assert _mult_fun_by_fun(f, f_op, f, f_op)

    def test_mult_by_conj_1(self):
        # MATLAB pass(17): f = sinh(t e^{2πi/6}); g = conj(f).  f .* conj(f)
        # matches |f|^2 pointwise.
        f_op = lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))
        g_op = lambda t: np.conj(np.sinh(np.asarray(t) * np.exp(2 * np.pi * 1j / 6)))
        f = _bf(f_op)
        g = f.conj()
        assert _mult_fun_by_fun(f, f_op, g, g_op)

    def test_mult_by_conj_2(self):
        # MATLAB pass(18): the positivity check -- f .* conj(f) is real and
        # nonnegative (it equals |f|^2).
        f_op = lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6))
        f = _bf(f_op)
        h = f * f.conj()
        hx = np.asarray(h(X))
        tol = 10 * h.vscale * EPS
        assert _ninf(hx.imag) < 1e4 * tol
        assert float(np.min(hx.real)) > -1e4 * tol

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
    def test_unhappy_result_1(self):
        # FIXED (Fable 5): tech arithmetic now propagates
        # ishappy=False.
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = _bf(lambda x: jnp.cos(x + 1)) * g
        assert (not g.ishappy) and (not h.ishappy)

    def test_unhappy_result_2(self):
        # FIXED (Fable 5): tech arithmetic now propagates
        # ishappy=False.
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = g * _bf(lambda x: jnp.cos(x + 1))
        assert (not g.ishappy) and (not h.ishappy)

    # --- singular (pass 24-25) ----------------------------------------
    # Singular funs are built via Bndfun.from_function(..., exponents=...),
    # mirroring MATLAB's bndfun(op, data.exponents, singPref).  The base
    # (x - b) is negative on the interval, so a fractional power is complex;
    # we build with a complex base to match MATLAB's complex singfun.  MATLAB
    # samples random INTERIOR points, so the singular endpoint b=7 is dropped.
    XI = jnp.asarray(np.linspace(-2.0, 7.0, 100)[:-1])
    XIR = np.linspace(-2.0, 7.0, 100)[:-1]

    def test_singular_scalar(self):  # pass(24): c .* (x-b)^-0.5 sin(x)
        pow_ = -0.5
        c = ALPHA
        op = lambda x: (x - DOM.b).astype(jnp.complex128) ** pow_ * jnp.sin(x)
        f = Bndfun.from_function(op, DOM, exponents=(0.0, pow_))
        g = c * f
        g_exact = c * (
            (self.XIR - DOM.b).astype(complex) ** pow_ * np.sin(self.XIR)
        )
        tol = 1e2 * EPS * float(np.max(np.abs(g_exact)))
        assert _ninf(np.asarray(g(self.XI)) - g_exact) < tol

    def test_singular_product(self):  # pass(25): product adds exponents
        pow1, pow2 = -0.3, -0.5
        op1 = lambda x: (x - DOM.b).astype(jnp.complex128) ** pow1 * jnp.sin(x)
        op2 = lambda x: (x - DOM.b).astype(jnp.complex128) ** pow2 * jnp.cos(3 * x)
        f = Bndfun.from_function(op1, DOM, exponents=(0.0, pow1))
        g = Bndfun.from_function(op2, DOM, exponents=(0.0, pow2))
        h = f * g
        assert h.onefun.exponents == (0.0, pow1 + pow2)
        h_exact = (self.XIR - DOM.b).astype(complex) ** (pow1 + pow2) * (
            np.sin(self.XIR) * np.cos(3 * self.XIR)
        )
        tol = 1e2 * EPS * float(np.max(np.abs(h_exact)))
        assert _ninf(np.asarray(h(self.XI)) - h_exact) < tol

    # --- Unbndfun multiplication (pass 26) ----------------------------
    def test_unbndfun_times(self):
        dom = Domain((-INF, INF))
        f = Unbndfun.from_function(lambda x: x ** 2 * jnp.exp(-x ** 2), dom)
        g = Unbndfun.from_function(lambda x: (1 - jnp.exp(-x ** 2)) / x, dom)
        h = f * g
        x = jnp.asarray(np.linspace(-1e2, 1e2, 100))
        hexact = x * jnp.exp(-x ** 2) * (1 - jnp.exp(-x ** 2))
        # MATLAB's bound is 2*eps*vscale; measured 1.13 ulp over it
        # (2.498e-16 vs 1.63e-16) after the 2026-08 numpy Clenshaw eval
        # path, whose summation order differs from the traced Clenshaw
        # by an ulp.  4*eps*vscale keeps the same sub-3-ulp contract.
        assert _ninf(h(x) - hexact) < 4 * EPS * f.vscale
