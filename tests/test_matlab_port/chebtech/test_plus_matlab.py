"""Port of MATLAB Chebfun tests/chebtech/test_plus.m (Opus 4.8).

Self-validating: each sum is checked against an analytic exact at the SAME
tolerance MATLAB uses.  The MATLAB file loops ``for n = 1:2`` over
``{chebtech1(), chebtech2()}``; every method is parametrized over both classes.

MATLAB ``testclass.make(@(x)f, [], pref)`` -> ``Tech.from_function(f)``.
MATLAB ``f + alpha`` / ``f + g`` -> Python ``f + alpha`` / ``f + g``.
MATLAB ``isequal(g1, g2)`` -> chebfunjax has no ``isequal``; the substitution
compares Chebyshev coefficients after zero-padding to a common length
(commutative float ops agree exactly, so the difference is 0).

Gaps vs MATLAB (honest xfail/skip), reported in the final summary:
- Chebtech1 cannot represent complex-valued functions (vals2coeffs/coeffs2vals
  drop the imaginary part); complex ``g_op`` sub-cases skip Chebtech1.
- Chebtech1.__add__ drops the imaginary part of a complex scalar addend.
- chebfunjax arithmetic (from_coeffs) always sets ishappy=True; happiness is
  not propagated through plus (the "unhappy result" checks xfail).
- array-valued / empty / dimension-error assertions have no scalar-tech analog.

Provenance
----------
MATLAB source : tests/chebtech/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
# arbitrary complex additive constant (matches test_plus.m)
ALPHA = -0.194758928283640 + 0.075474485412665j

_CT1_COMPLEX = (
    "Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it cannot "
    "represent complex-valued functions built via from_function"
)
_CT1_ADD = (
    "Chebtech1.__add__/__sub__ does not promote the coeff dtype, so it drops "
    "the imaginary part of a complex scalar addend"
)
_UNHAPPY = (
    "chebfunjax Chebtech arithmetic builds results via from_coeffs, which "
    "always sets ishappy=True; happiness is not propagated through plus"
)
_ARRAY = "chebfunjax Chebtech is scalar-valued; no array-valued/quasimatrix techs"
_EMPTY = "chebfunjax has no empty-tech / array-concat API for isempty checks"
_DIMERR = "chebfunjax has no MATLAB dimension-mismatch error identifier"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _coeff_diff(f, g):
    """||coeffs(f) - coeffs(g)||_inf after zero-padding to a common length."""
    a = jnp.asarray(f.coeffs)
    b = jnp.asarray(g.coeffs)
    n = max(a.shape[0], b.shape[0])
    dt = jnp.result_type(a.dtype, b.dtype)
    ap = jnp.zeros(n, dt).at[: a.shape[0]].set(a)
    bp = jnp.zeros(n, dt).at[: b.shape[0]].set(b)
    return _ninf(ap - bp)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechPlus:
    def test_empty_arguments(self, Tech):
        # pass(n, 1): isempty(f + f) etc. with an empty f.
        pytest.skip(_EMPTY)

    # -- Addition with a (complex) scalar. pass(n, 2:3). --
    def test_add_scalar_commutes(self, Tech):
        # pass(n, 2): isequal(f + alpha, alpha + f).
        f = Tech.from_function(lambda x: jnp.sin(x))
        g1 = f + ALPHA
        g2 = ALPHA + f
        assert _coeff_diff(g1, g2) == 0.0

    def test_add_scalar_matches_exact(self, Tech):
        # pass(n, 3): feval(f + alpha) == sin + alpha.
        if Tech is Chebtech1:
            pytest.xfail(_CT1_ADD)
        f = Tech.from_function(lambda x: jnp.sin(x))
        g1 = f + ALPHA
        err = _ninf(g1(X) - (jnp.sin(X) + ALPHA))
        assert err <= 10 * g1.vscale * EPS

    # -- Addition of two chebtech objects. pass(n, 4:11). --
    def test_add_zero_functions_commute(self, Tech):
        # pass(n, 4): isequal(f + f, f + f) for f == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        assert _coeff_diff(f + f, f + f) == 0.0

    def test_add_zero_functions_exact(self, Tech):
        # pass(n, 5): feval(f + f) == 0 for f == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        h = f + f
        assert _ninf(h(X)) <= 1e4 * h.vscale * EPS

    def test_add_exp_and_lorentzian_commute(self, Tech):
        # pass(n, 6): isequal(f + g, g + f), f = e^x - 1, g = 1/(1+x^2).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        assert _coeff_diff(f + g, g + f) == 0.0

    def test_add_exp_and_lorentzian_exact(self, Tech):
        # pass(n, 7): feval(f + g) == (e^x - 1) + 1/(1+x^2).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        h = f + g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) + 1.0 / (1 + X ** 2)))
        assert err <= 1e4 * h.vscale * EPS

    def test_add_exp_and_high_freq_commute(self, Tech):
        # pass(n, 8): isequal(f + g, g + f), g = cos(1e4 x).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        assert _coeff_diff(f + g, g + f) == 0.0

    def test_add_exp_and_high_freq_exact(self, Tech):
        # pass(n, 9): feval(f + g) == (e^x - 1) + cos(1e4 x).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        h = f + g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) + jnp.cos(1e4 * X)))
        assert err <= 1e4 * h.vscale * EPS

    def test_add_exp_and_complex_sinh_commute(self, Tech):
        # pass(n, 10): isequal(f + g, g + f), g = sinh(t e^{2pi i/6}).
        if Tech is Chebtech1:
            pytest.skip(_CT1_COMPLEX)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        assert _coeff_diff(f + g, g + f) == 0.0

    def test_add_exp_and_complex_sinh_exact(self, Tech):
        # pass(n, 11): feval(f + g) == (e^x - 1) + sinh(t e^{2pi i/6}).
        if Tech is Chebtech1:
            pytest.skip(_CT1_COMPLEX)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f + g
        exact = (jnp.exp(X) - 1) + jnp.sinh(X * jnp.exp(2j * jnp.pi / 6))
        assert _ninf(h(X) - exact) <= 1e4 * h.vscale * EPS

    def test_array_valued_zero_sum(self, Tech):
        # pass(n, 12:13): array-valued zero + zero.
        pytest.skip(_ARRAY)

    def test_array_valued_plus_scalar(self, Tech):
        # pass(n, 14:15): [sin cos exp] + alpha.
        pytest.skip(_ARRAY)

    def test_array_valued_sum(self, Tech):
        # pass(n, 16:17): [sin cos exp] + [cosh airy(ix) sinh].
        pytest.skip(_ARRAY)

    def test_dimension_mismatch_error(self, Tech):
        # pass(n, 18): array-valued + scalar-valued -> dimension error.
        pytest.skip(_DIMERR)

    def test_plus_matches_direct_construction(self, Tech):
        # pass(n, 19): coeffs of (f + g) match direct construction, tol 10*eps.
        f = Tech.from_function(lambda x: x)
        g = Tech.from_function(lambda x: jnp.cos(x) - 1)
        h1 = f + g
        h2 = Tech.from_function(lambda x: x + jnp.cos(x) - 1)
        assert _coeff_diff(h1, h2) < 10 * EPS

    def test_unhappy_plus_happy_stays_unhappy(self, Tech):
        # pass(n, 20): f (happy) + g (unhappy) -> unhappy.
        pytest.xfail(_UNHAPPY)

    def test_happy_plus_unhappy_stays_unhappy(self, Tech):
        # pass(n, 21): g (unhappy) + f (happy) -> unhappy.
        pytest.xfail(_UNHAPPY)

    def test_array_valued_scalar_row_expansion(self, Tech):
        # pass(n, 22): [sin cos exp] + [1 2 3].
        pytest.skip(_ARRAY)

    def test_scalar_expansion_in_chebtech(self, Tech):
        # pass(n, 23): sin(x) + [1 2 3] (scalar expanded to array-valued).
        pytest.skip(_ARRAY)
