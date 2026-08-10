"""Port of MATLAB Chebfun tests/classicfun/test_minus.m (Fable 5).

Self-validating: subtraction (fun-scalar, fun-fun) is checked against the
analytic difference at the SAME tolerances MATLAB uses.  ``isequal`` is
reproduced by comparing the underlying Chebyshev coefficients and domains
(a faithful equivalent of @classicfun/isequal.m).  MATLAB uses a random
complex ``alpha``; any complex constant exercises the same code path, so a
fixed one is used here.

All 22 MATLAB assertions are covered: scalar, array-valued (3-column),
complex and Unbndfun subtraction, plus MATLAB >= 9.1 implicit expansion of
an array-valued fun by a scalar-valued one (pass 18).  No gaps.

Provenance
----------
MATLAB source : tests/classicfun/test_minus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special as sp

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
    def test_empty(self):
        # pass(1): isempty(f - f) && isempty(f - g) && isempty(g - f)
        f = Bndfun.empty()
        g = Bndfun.from_function(jnp.sin, DOM)
        assert (f - f).isempty()
        assert (f - g).isempty()
        assert (g - f).isempty()

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
    # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun arithmetic.
    _ZEROS = staticmethod(lambda x: jnp.stack([jnp.zeros_like(x)] * 3, axis=-1))
    _SCE = staticmethod(
        lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
    )
    _CAS = staticmethod(
        lambda x: jnp.stack(
            [jnp.cosh(x), jnp.asarray(sp.airy(1j * np.asarray(x))[0]), jnp.sinh(x)],
            axis=-1,
        )
    )

    def test_array_zeros_isequal(self):
        # pass(12): zeros(3col) - zeros(3col): isequal(f-f, -(f-f)).
        f = _bf(self._ZEROS)
        assert _isequal(f - f, -(f - f))

    def test_array_zeros_norm(self):
        # pass(13): value of zeros - zeros is ~0.
        f = _bf(self._ZEROS)
        h1 = f - f
        assert _ninf(h1(X)) <= 100 * max(h1.vscale, 1.0) * EPS

    def test_array_scalar_isequal(self):
        # pass(14): [sin cos exp] - alpha: isequal(f-alpha, -(alpha-f)).
        f = _bf(self._SCE)
        assert _isequal(f - ALPHA, -(ALPHA - f))

    def test_array_scalar_norm(self):
        # pass(15): value of [sin cos exp] - alpha, tol 10*vscale*eps.
        f = _bf(self._SCE)
        g1 = f - ALPHA
        gexact = self._SCE(X) - ALPHA
        assert _ninf(g1(X) - gexact) < 10 * g1.vscale * EPS

    def test_array_function_isequal(self):
        # pass(16): [sin cos exp] - [cosh airy(1i x) sinh]: isequal(h1, -(g-f)).
        f = _bf(self._SCE)
        g = _bf(self._CAS)
        assert _isequal(f - g, -(g - f))

    def test_array_function_norm(self):
        # pass(17): value of the array difference, tol 100*vscale*eps.
        f = _bf(self._SCE)
        g = _bf(self._CAS)
        h1 = f - g
        hexact = self._SCE(X) - self._CAS(X)
        assert _ninf(h1(X) - hexact) <= 100 * h1.vscale * EPS

    def test_implicit_expansion(self):
        # pass(18): f(3-col) - g(scalar-col).  On MATLAB >= 9.1 this succeeds
        # via implicit expansion (pass(18) = true); only pre-9.1 MATLAB raised
        # 'Matrix dimensions must agree.'.  chebfunjax broadcasts the same way.
        f = _bf(self._SCE)
        g = _bf(jnp.sin)
        h = f - g
        assert np.asarray(h(X)).shape == (X.shape[0], 3)
        hexact = self._SCE(X) - jnp.sin(X)[:, None]
        assert _ninf(h(X) - hexact) <= 100 * h.vscale * EPS
        # Reversed order broadcasts to the negation.
        assert _ninf((g - f)(X) + hexact) <= 100 * h.vscale * EPS

    def test_column_count_mismatch_raises(self):
        # Not a MATLAB assertion: a genuinely incompatible column count (3 - 2)
        # is not broadcastable and must still be rejected.
        f = _bf(self._SCE)
        g = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        with pytest.raises((TypeError, ValueError)):
            f - g

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
