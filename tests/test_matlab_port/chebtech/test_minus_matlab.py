"""Port of MATLAB Chebfun tests/chebtech/test_minus.m (Opus 4.8).

Self-validating: each difference is checked against an analytic exact at the
SAME tolerance MATLAB uses.  The MATLAB file loops ``for n = 1:2`` over
``{chebtech1(), chebtech2()}``; every method is parametrized over both classes.

MATLAB ``isequal(g1, -g2)`` -> chebfunjax has no ``isequal``; the substitution
compares Chebyshev coefficients after zero-padding to a common length.

Gaps vs MATLAB (honest xfail/skip), reported in the final summary:
- Chebtech1 cannot represent complex-valued functions (vals2coeffs/coeffs2vals
  drop the imaginary part); complex ``g_op`` sub-cases skip Chebtech1.
- Chebtech1.__sub__ drops the imaginary part of a complex scalar operand.
- chebfunjax arithmetic (from_coeffs) always sets ishappy=True; happiness is
  not propagated through minus (the "unhappy result" checks xfail).
- empty / dimension-error assertions have no scalar-tech analog.

Array-valued: Chebtech now supports (n, m) coefficient matrices, so the
array-valued minus cases (pass 12:17) are ported.  Chebtech1 still drops the
imaginary part of a complex *scalar* operand, so the array-valued exact check
with the complex ALPHA (pass 15) xfails on Chebtech1 just like pass 3.

Provenance
----------
MATLAB source : tests/chebtech/test_minus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.special

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
# arbitrary complex constant (test_minus.m uses randn() + 1i*randn()).
ALPHA = -0.912132456789012 + 0.632984567890123j


def _airy_ix(x):
    """MATLAB ``airy(1i*x)`` — Airy Ai on the imaginary axis (complex-valued)."""
    return jnp.asarray(scipy.special.airy(1j * np.asarray(x))[0])

_CT1_COMPLEX = (
    "Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it cannot "
    "represent complex-valued functions built via from_function"
)
_CT1_SUB = (
    "Chebtech1.__sub__/__add__ does not promote the coeff dtype, so it drops "
    "the imaginary part of a complex scalar operand"
)
_UNHAPPY = (
    "chebfunjax Chebtech arithmetic builds results via from_coeffs, which "
    "always sets ishappy=True; happiness is not propagated through minus"
)
_EMPTY = "chebfunjax has no empty-tech / array-concat API for isempty checks"
_DIMERR = "chebfunjax has no MATLAB dimension-mismatch error identifier"


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _coeff_diff(f, g):
    """||coeffs(f) - coeffs(g)||_inf after zero-padding to a common length.

    Handles both scalar (n,) and array-valued (n, m) coefficient arrays.
    """
    a = jnp.asarray(f.coeffs)
    b = jnp.asarray(g.coeffs)
    n = max(a.shape[0], b.shape[0])
    dt = jnp.result_type(a.dtype, b.dtype)
    ap = jnp.zeros((n,) + a.shape[1:], dt).at[: a.shape[0]].set(a)
    bp = jnp.zeros((n,) + b.shape[1:], dt).at[: b.shape[0]].set(b)
    return _ninf(ap - bp)


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechMinus:
    def test_empty_arguments(self, Tech):
        # pass(n, 1): isempty(f - f) etc. with an empty f.
        pytest.skip(_EMPTY)

    # -- Subtraction with a (complex) scalar. pass(n, 2:3). --
    def test_sub_scalar_negation(self, Tech):
        # pass(n, 2): isequal(f - alpha, -(alpha - f)).
        f = Tech.from_function(lambda x: jnp.sin(x))
        g1 = f - ALPHA
        g2 = ALPHA - f
        assert _coeff_diff(g1, -g2) == 0.0

    def test_sub_scalar_matches_exact(self, Tech):
        # pass(n, 3): feval(f - alpha) == sin - alpha.
        if Tech is Chebtech1:
            pytest.xfail(_CT1_SUB)
        f = Tech.from_function(lambda x: jnp.sin(x))
        g1 = f - ALPHA
        err = _ninf(g1(X) - (jnp.sin(X) - ALPHA))
        assert err <= 10 * g1.vscale * EPS

    # -- Subtraction of two chebtech objects. pass(n, 4:11). --
    def test_sub_zero_functions_negation(self, Tech):
        # pass(n, 4): isequal(f - f, -(f - f)) for f == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        assert _coeff_diff(f - f, -(f - f)) == 0.0

    def test_sub_zero_functions_exact(self, Tech):
        # pass(n, 5): feval(f - f) == 0 for f == 0.
        f = Tech.from_function(lambda x: jnp.zeros_like(x))
        h = f - f
        assert _ninf(h(X)) <= 1e4 * h.vscale * EPS

    def test_sub_exp_and_lorentzian_negation(self, Tech):
        # pass(n, 6): isequal(f - g, -(g - f)), f = e^x - 1, g = 1/(1+x^2).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        assert _coeff_diff(f - g, -(g - f)) == 0.0

    def test_sub_exp_and_lorentzian_exact(self, Tech):
        # pass(n, 7): feval(f - g) == (e^x - 1) - 1/(1+x^2).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        h = f - g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) - 1.0 / (1 + X ** 2)))
        assert err <= 1e4 * h.vscale * EPS

    def test_sub_exp_and_high_freq_negation(self, Tech):
        # pass(n, 8): isequal(f - g, -(g - f)), g = cos(1e4 x).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        assert _coeff_diff(f - g, -(g - f)) == 0.0

    def test_sub_exp_and_high_freq_exact(self, Tech):
        # pass(n, 9): feval(f - g) == (e^x - 1) - cos(1e4 x).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        h = f - g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) - jnp.cos(1e4 * X)))
        assert err <= 1e4 * h.vscale * EPS

    def test_sub_exp_and_complex_sinh_negation(self, Tech):
        # pass(n, 10): isequal(f - g, -(g - f)), g = sinh(t e^{2pi i/6}).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        assert _coeff_diff(f - g, -(g - f)) == 0.0

    def test_sub_exp_and_complex_sinh_exact(self, Tech):
        # pass(n, 11): feval(f - g) == (e^x - 1) - sinh(t e^{2pi i/6}).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f - g
        exact = (jnp.exp(X) - 1) - jnp.sinh(X * jnp.exp(2j * jnp.pi / 6))
        assert _ninf(h(X) - exact) <= 1e4 * h.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): array-valued zero - zero.
    def test_array_valued_zero_diff_negation(self, Tech):
        # pass(n, 12): isequal(f - f, -(f - f)) for array-valued f == [0 0 0].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)], axis=-1
            )
        )
        assert _coeff_diff(f - f, -(f - f)) == 0.0

    def test_array_valued_zero_diff_exact(self, Tech):
        # pass(n, 13): feval(f - f) == 0 for array-valued f == [0 0 0].
        f = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.zeros_like(x), jnp.zeros_like(x), jnp.zeros_like(x)], axis=-1
            )
        )
        h = f - f
        assert _ninf(h(X)) <= 1e4 * h.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos exp] - alpha.
    def test_array_valued_minus_scalar_negation(self, Tech):
        # pass(n, 14): isequal(f - alpha, -(alpha - f)), f = [sin cos exp].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g1 = f - ALPHA
        g2 = ALPHA - f
        assert _coeff_diff(g1, -g2) == 0.0

    def test_array_valued_minus_scalar_exact(self, Tech):
        # pass(n, 15): feval([sin cos exp] - alpha) == exact.
        if Tech is Chebtech1:
            pytest.xfail(_CT1_SUB)
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g1 = f - ALPHA
        exact = jnp.stack([jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1) - ALPHA
        assert _ninf(g1(X) - exact) <= 10 * g1.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos exp] - [cosh airy(ix) sinh].
    def test_array_valued_diff_negation(self, Tech):
        # pass(n, 16): isequal(f - g, -(g - f)), f = [sin cos exp], g = [cosh airy(1i x) sinh].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.cosh(x), _airy_ix(x), jnp.sinh(x)], axis=-1)
        )
        assert _coeff_diff(f - g, -(g - f)) == 0.0

    def test_array_valued_diff_exact(self, Tech):
        # pass(n, 17): feval(f - g) == [sin cos exp] - [cosh airy(1i x) sinh].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.cosh(x), _airy_ix(x), jnp.sinh(x)], axis=-1)
        )
        h = f - g
        exact = jnp.stack([jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1) - jnp.stack(
            [jnp.cosh(X), _airy_ix(X), jnp.sinh(X)], axis=-1
        )
        assert _ninf(h(X) - exact) <= 1e4 * h.vscale * EPS

    def test_dimension_mismatch_error(self, Tech):
        # pass(n, 18): array-valued - scalar-valued -> dimension error.
        pytest.skip(_DIMERR)

    def test_minus_matches_direct_construction(self, Tech):
        # pass(n, 19): (f - g) - direct construction has ~0 coeffs, tol 10*eps.
        f = Tech.from_function(lambda x: x)
        g = Tech.from_function(lambda x: jnp.cos(x) - 1)
        h1 = f - g
        h2 = Tech.from_function(lambda x: x - (jnp.cos(x) - 1))
        h3 = h1 - h2
        assert _ninf(h3.coeffs) < 10 * EPS

    def test_unhappy_minus_happy_stays_unhappy(self, Tech):
        # pass(n, 20): f (happy) - g (unhappy) -> unhappy.
        pytest.xfail(_UNHAPPY)

    def test_happy_minus_unhappy_stays_unhappy(self, Tech):
        # pass(n, 21): g (unhappy) - f (happy) -> unhappy.
        pytest.xfail(_UNHAPPY)
