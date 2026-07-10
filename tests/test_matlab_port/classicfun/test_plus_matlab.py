"""Port of MATLAB Chebfun tests/classicfun/test_plus.m (Opus 4.8).

Self-validating: addition (fun+scalar, fun+fun) is checked against the
analytic sum at the SAME tolerances MATLAB uses.  ``isequal`` is reproduced by
comparing the underlying Chebyshev coefficients and domains (a faithful
equivalent of @classicfun/isequal.m, which compares the onefun and the map).

Provenance
----------
MATLAB source : tests/classicfun/test_plus.m
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
ALPHA = -0.194758928283640 + 0.075474485412665j
INF = np.inf


def _bf(op):
    return Bndfun.from_function(op, DOM)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _isequal(f, g):
    """Faithful equivalent of classicfun/isequal: same domain + same coeffs."""
    if f.domain != g.domain:
        return False
    fc, gc = np.asarray(f.coeffs), np.asarray(g.coeffs)
    return fc.shape == gc.shape and np.array_equal(fc, gc)


class TestClassicfunPlus:
    @pytest.mark.xfail(
        reason="chebfunjax has no empty-fun construction (bndfun() with no "
        "args), so empty-argument addition cannot be exercised."
    )
    def test_empty(self):
        raise NotImplementedError("empty Bndfun")

    # --- add a scalar (alpha is complex) ------------------------------
    def test_add_scalar_isequal(self):
        f = _bf(jnp.sin)
        assert _isequal(f + ALPHA, ALPHA + f)

    def test_add_scalar_norm(self):
        f = _bf(jnp.sin)
        g1 = f + ALPHA
        gexact = jnp.sin(X) + ALPHA
        assert _ninf(g1(X) - gexact) < 10 * g1.vscale * EPS

    # --- add two funs: zeros + zeros ----------------------------------
    def test_add_zeros_isequal(self):
        f = _bf(lambda x: jnp.zeros_like(x))
        assert _isequal(f + f, f + f)

    def test_add_zeros_norm(self):
        f = _bf(lambda x: jnp.zeros_like(x))
        h1 = f + f
        assert _ninf(h1(X)) <= 100 * max(h1.vscale, 1.0) * EPS

    # --- (exp(x)-1) + 1/(1+x^2) ---------------------------------------
    def test_add_exp_recip_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: 1.0 / (1 + x ** 2))
        assert _isequal(f + g, g + f)

    def test_add_exp_recip_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: 1.0 / (1 + x ** 2))
        h1 = f + g
        hexact = (jnp.exp(X) - 1) + 1.0 / (1 + X ** 2)
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    # --- (exp(x)-1) + cos(1e4 x) --------------------------------------
    def test_add_cos1e4_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: jnp.cos(1e4 * x))
        assert _isequal(f + g, g + f)

    def test_add_cos1e4_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda x: jnp.cos(1e4 * x))
        h1 = f + g
        hexact = (jnp.exp(X) - 1) + jnp.cos(1e4 * X)
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    # --- (exp(x)-1) + sinh(t*exp(2*pi*i/6)) (complex) -----------------
    def test_add_sinh_complex_isequal(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6)))
        assert _isequal(f + g, g + f)

    def test_add_sinh_complex_norm(self):
        f = _bf(lambda x: jnp.exp(x) - 1)
        g = _bf(lambda t: jnp.sinh(t * np.exp(2 * np.pi * 1j / 6)))
        h1 = f + g
        hexact = (jnp.exp(X) - 1) + jnp.sinh(X * np.exp(2 * np.pi * 1j / 6))
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
        reason="chebfunjax lacks array-valued Bndfun: adding a 3-column fun and "
        "a scalar-valued fun should raise a dimension-mismatch error."
    )
    def test_dimension_mismatch(self):
        raise NotImplementedError

    # --- direct construction vs plus ----------------------------------
    def test_direct_construction(self):
        f = _bf(lambda x: x)
        g = _bf(lambda x: jnp.cos(x) - 1)
        h1 = f + g
        h2 = _bf(lambda x: x + jnp.cos(x) - 1)
        assert _ninf(h1(X) - h2(X)) < 2 * (10 * EPS)

    # --- happy + unhappy ----------------------------------------------
    @pytest.mark.xfail(
        reason="chebfunjax arithmetic does not propagate ishappy=False from an "
        "unhappy operand: sqrt(x+1) is unhappy but cos(x+1)+sqrt(x+1) reports "
        "ishappy=True."
    )
    def test_unhappy_plus_happy(self):
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = _bf(lambda x: jnp.cos(x + 1)) + g
        assert (not g.ishappy) and (not h.ishappy)

    @pytest.mark.xfail(
        reason="chebfunjax arithmetic does not propagate ishappy=False."
    )
    def test_happy_plus_unhappy(self):
        g = _bf(lambda x: jnp.sqrt(x + 1))
        h = g + _bf(lambda x: jnp.cos(x + 1))
        assert (not g.ishappy) and (not h.ishappy)

    # --- singular (pass 22) -------------------------------------------
    @pytest.mark.xfail(
        reason="chebfunjax lacks singular (exponents [0 -1]) Bndfun."
    )
    def test_singular(self):
        raise NotImplementedError("singular Bndfun plus")

    # --- Unbndfun addition (pass 23) ----------------------------------
    def test_unbndfun_plus(self):
        f = Unbndfun.from_function(lambda x: jnp.exp(-x ** 2), Domain((-INF, INF)))
        g = Unbndfun.from_function(
            lambda x: x ** 2 * jnp.exp(-x ** 2), Domain((-INF, INF))
        )
        h = f + g
        x = jnp.asarray(np.linspace(-1e2, 1e2, 100))
        hexact = jnp.exp(-x ** 2) + x ** 2 * jnp.exp(-x ** 2)
        assert _ninf(h(x) - hexact) < 1e1 * EPS * h.vscale
