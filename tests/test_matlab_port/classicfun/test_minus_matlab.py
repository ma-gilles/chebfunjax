"""Port of MATLAB Chebfun tests/classicfun/test_minus.m (Opus 4.8).

Self-validating: subtraction (fun-scalar, fun-fun) is checked against the
analytic difference at the SAME tolerances MATLAB uses.  ``isequal`` is
reproduced by comparing the underlying Chebyshev coefficients and domains
(a faithful equivalent of @classicfun/isequal.m).  MATLAB uses a random
complex ``alpha``; any complex constant exercises the same code path, so a
fixed one is used here.

Provenance
----------
MATLAB source : tests/classicfun/test_minus.m
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
X = jnp.asarray(np.linspace(-2.0, 7.0, 100))
ALPHA = 0.847113928283640 - 1.234474485412665j  # stand-in for randn()+i*randn()
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


class TestClassicfunMinus:
    @pytest.mark.xfail(
        reason="chebfunjax has no empty-fun construction (bndfun() with no "
        "args)."
    )
    def test_empty(self):
        raise NotImplementedError("empty Bndfun")

    # --- subtract a scalar --------------------------------------------
    def test_sub_scalar_isequal(self):
        f = _bf(jnp.sin)
        g1 = f - ALPHA
        g2 = ALPHA - f
        assert _isequal(g1, -g2)

    def test_sub_scalar_norm(self):
        f = _bf(jnp.sin)
        g1 = f - ALPHA
        gexact = jnp.sin(X) - ALPHA
        assert _ninf(g1(X) - gexact) < 10 * f.vscale * EPS

    # --- subtract two funs: zeros - zeros -----------------------------
    def test_sub_zeros_isequal(self):
        f = _bf(lambda x: jnp.zeros_like(x))
        assert _isequal(f - f, -(f - f))

    def test_sub_zeros_norm(self):
        f = _bf(lambda x: jnp.zeros_like(x))
        h1 = f - f
        assert _ninf(h1(X)) <= 100 * max(h1.vscale, 1.0) * EPS

    # --- (exp(x)-1) - 1/(1+x^2) ---------------------------------------
    def test_sub_exp_recip_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: 1.0 / (1 + x ** 2))
        assert _isequal(f - g, -(g - f))

    def test_sub_exp_recip_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: 1.0 / (1 + x ** 2))
        h1 = f - g
        hexact = (jnp.exp(X) - 1) - 1.0 / (1 + X ** 2)
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    # --- (exp(x)-1) - cos(1e4 x) --------------------------------------
    def test_sub_cos1e4_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: jnp.cos(1e4 * x))
        assert _isequal(f - g, -(g - f))

    def test_sub_cos1e4_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: jnp.cos(1e4 * x))
        h1 = f - g
        hexact = (jnp.exp(X) - 1) - jnp.cos(1e4 * X)
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    # --- (exp(x)-1) - sinh(t*exp(2*pi*i/6)) (complex) -----------------
    def test_sub_sinh_complex_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6)))
        assert _isequal(f - g, -(g - f))

    def test_sub_sinh_complex_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6)))
        h1 = f - g
        hexact = (jnp.exp(X) - 1) - jnp.sinh(X * np.exp(2 * np.pi * 1j / 6))
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    # --- array-valued cases (pass 12-17) ------------------------------
    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_zeros_isequal(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_zeros_norm(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_scalar_isequal(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_scalar_norm(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_function_isequal(self):
        raise NotImplementedError

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued Bndfun.")
    def test_array_function_norm(self):
        raise NotImplementedError

    @pytest.mark.xfail(
        reason="chebfunjax lacks array-valued Bndfun: subtracting a 3-column "
        "fun and a scalar-valued fun should raise a dimension-mismatch error."
    )
    def test_dimension_mismatch(self):
        raise NotImplementedError

    # --- direct construction vs minus ---------------------------------
    def test_direct_construction(self):
        f = _bf(lambda x: x)
        g = _bf(lambda x: jnp.cos(x) - 1)
        h1 = f - g
        h2 = _bf(lambda x: x - (jnp.cos(x) - 1))
        assert _ninf(h1(X) - h2(X)) < 1e1 * EPS * h1.vscale

    # --- happy - unhappy ----------------------------------------------
    def test_unhappy_result_1(self):
        # FIXED (Fable 5): tech arithmetic now propagates
        # ishappy=False.
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = _bf(lambda x: jnp.cos(x + 1)) - g
        assert (not g.ishappy) and (not h.ishappy)

    def test_unhappy_result_2(self):
        # FIXED (Fable 5): tech arithmetic now propagates
        # ishappy=False.
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = g - _bf(lambda x: jnp.cos(x + 1))
        assert (not g.ishappy) and (not h.ishappy)

    # --- Unbndfun subtraction (pass 22) -------------------------------
    def test_unbndfun_minus(self):
        dom = Domain((-INF, 3 * np.pi))
        f = Unbndfun.from_function(lambda x: x * jnp.exp(x), dom)
        g = Unbndfun.from_function(lambda x: (1 - jnp.exp(x)) / x, dom)
        h = f - g
        x = jnp.asarray(np.linspace(-1e6, 3 * np.pi, 100))
        hexact = x * jnp.exp(x) - (1 - jnp.exp(x)) / x
        assert _ninf(h(x) - hexact) < 1e1 * EPS * h.vscale
