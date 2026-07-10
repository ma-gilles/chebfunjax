"""Port of MATLAB Chebfun tests/chebtech/test_times.m (Opus 4.8).

Self-validating: each pointwise product is checked against an analytic exact at
the SAME tolerance MATLAB uses.  MATLAB ``.*`` -> Python ``*``.  The file loops
``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; every method is
parametrized over both classes.

MATLAB ``isequal(g1, g2)`` -> chebfunjax has no ``isequal``; the substitution
compares Chebyshev coefficients after zero-padding to a common length.

Gaps vs MATLAB (honest xfail/skip), reported in the final summary:
- Chebtech1 cannot represent complex-valued functions (vals2coeffs/coeffs2vals
  drop the imaginary part); complex operands skip Chebtech1.
- chebfunjax has no ``conj`` on Chebtech; the ``f .* conj(f)`` cases skip.
- chebfunjax arithmetic (from_coeffs) always sets ishappy=True; happiness is
  not propagated through times (the "unhappy result" checks xfail).
- array-valued / empty / dimension-error assertions have no scalar-tech analog.

Provenance
----------
MATLAB source : tests/chebtech/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
# arbitrary complex multiplicative constant (matches test_times.m).
ALPHA = -0.194758928283640 + 0.075474485412665j

_CT1_COMPLEX = (
    "Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it cannot "
    "represent complex-valued functions built via from_function"
)
_UNHAPPY = (
    "chebfunjax Chebtech arithmetic builds results via from_coeffs, which "
    "always sets ishappy=True; happiness is not propagated through times"
)
_CONJ = "chebfunjax Chebtech has no conj method"
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
class TestChebtechTimes:
    def test_empty_arguments(self, Tech):
        # pass(n, 1): isempty(f .* f) etc. with an empty f.
        pytest.skip(_EMPTY)

    # -- Multiplication by a (complex) scalar. pass(n, 2:3). --
    def test_mult_scalar_commutes(self, Tech):
        # pass(n, 2): isequal(f .* alpha, alpha .* f).
        f = Tech.from_function(lambda x: jnp.sin(x))
        assert _coeff_diff(f * ALPHA, ALPHA * f) == 0.0

    def test_mult_scalar_matches_exact(self, Tech):
        # pass(n, 3): feval(f .* alpha) == sin * alpha.
        f = Tech.from_function(lambda x: jnp.sin(x))
        g1 = f * ALPHA
        err = _ninf(g1(X) - (jnp.sin(X) * ALPHA))
        assert err < 10 * g1.vscale * EPS

    def test_array_valued_mult_scalar(self, Tech):
        # pass(n, 4:5): [sin cos] .* alpha.
        pytest.skip(_ARRAY)

    # -- Multiplication by constant functions. pass(n, 6). --
    def test_mult_by_constant_function(self, Tech):
        # pass(n, 6): sin .* (alpha * ones).
        if Tech is Chebtech1:
            pytest.skip(_CT1_COMPLEX)
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = Tech.from_function(lambda x: ALPHA * jnp.ones_like(x))
        h = f * g
        err = _ninf(h(X) - (jnp.sin(X) * ALPHA))
        assert err < 1e4 * h.vscale * EPS

    def test_array_valued_mult_constants(self, Tech):
        # pass(n, 7): [sin cos] .* [alpha beta] (constant columns).
        pytest.skip(_ARRAY)

    # -- Spot-checks of two-chebtech products. pass(n, 8:11). --
    def test_mult_ones_by_ones(self, Tech):
        # pass(n, 8): ones .* ones.
        f = Tech.from_function(lambda x: jnp.ones_like(x))
        h = f * f
        assert _ninf(h(X) - jnp.ones_like(X)) < 1e4 * h.vscale * EPS

    def test_mult_exp_by_lorentzian(self, Tech):
        # pass(n, 9): (e^x - 1) .* 1/(1+x^2).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        h = f * g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) * (1.0 / (1 + X ** 2))))
        assert err < 1e4 * h.vscale * EPS

    def test_mult_exp_by_high_freq(self, Tech):
        # pass(n, 10): (e^x - 1) .* cos(1e4 x).
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        h = f * g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) * jnp.cos(1e4 * X)))
        assert err < 1e4 * h.vscale * EPS

    def test_mult_exp_by_complex_sinh(self, Tech):
        # pass(n, 11): (e^x - 1) .* sinh(t e^{2pi i/6}).
        if Tech is Chebtech1:
            pytest.skip(_CT1_COMPLEX)
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f * g
        exact = (jnp.exp(X) - 1) * jnp.sinh(X * jnp.exp(2j * jnp.pi / 6))
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    def test_array_valued_mult_scalar_tech(self, Tech):
        # pass(n, 12:13): [sin cos exp] .* tanh.
        pytest.skip(_ARRAY)

    def test_array_valued_mult_array(self, Tech):
        # pass(n, 14): [sin cos exp] .* [sinh cosh tanh].
        pytest.skip(_ARRAY)

    def test_array_valued_dim_mismatch(self, Tech):
        # pass(n, 15): array-valued .* with mismatched columns -> dim2 error.
        pytest.skip(_DIMERR)

    # -- Specially handled cases (positivity adjustments). pass(n, 16:20). --
    def test_mult_complex_sinh_squared(self, Tech):
        # pass(n, 16): sinh(t e^{2pi i/6}) .* itself.
        if Tech is Chebtech1:
            pytest.skip(_CT1_COMPLEX)
        f = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f * f
        exact = jnp.sinh(X * jnp.exp(2j * jnp.pi / 6)) ** 2
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    def test_mult_by_conjugate_accuracy(self, Tech):
        # pass(n, 17): f .* conj(f).
        pytest.skip(_CONJ)

    def test_mult_by_conjugate_positivity(self, Tech):
        # pass(n, 18): f .* conj(f) is nonnegative.
        pytest.skip(_CONJ)

    def test_mult_real_square_accuracy(self, Tech):
        # pass(n, 19): (e^x - 1) .* (e^x - 1) accuracy.
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        h = f * f
        err = _ninf(h(X) - (jnp.exp(X) - 1) ** 2)
        assert err < 1e4 * h.vscale * EPS

    def test_mult_real_square_positivity(self, Tech):
        # pass(n, 20): (e^x - 1)^2 is nonnegative on the Chebyshev grid.
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        h = f * f
        tol = 1e4 * h.vscale * EPS
        values = jnp.real(Tech.coeffs2vals(h.coeffs))
        assert bool(jnp.all(values >= -tol))

    def test_mult_matches_direct_construction(self, Tech):
        # pass(n, 21): coeffs of (f .* g) match direct construction, tol 50*eps.
        # MATLAB prolongs h2 to length(h1); _coeff_diff zero-pads to a common
        # length, which is equivalent for the inf-norm comparison.
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        h1 = f * g
        h2 = Tech.from_function(lambda x: (jnp.exp(x) - 1) * (1.0 / (1 + x ** 2)))
        assert _coeff_diff(h1, h2) < 50 * EPS

    def test_array_valued_moderate_dims(self, Tech):
        # pass(n, 22): [f...] .* [f...] for moderate array dimensions.
        pytest.skip(_ARRAY)

    def test_unhappy_times_happy_stays_unhappy(self, Tech):
        # pass(n, 23): f (happy) .* g (unhappy) -> unhappy.
        pytest.xfail(_UNHAPPY)

    def test_happy_times_unhappy_stays_unhappy(self, Tech):
        # pass(n, 24): g (unhappy) .* f (happy) -> unhappy.
        pytest.xfail(_UNHAPPY)

    def test_polynomial_product_degree(self, Tech):
        # pass(n, 25): products give the correct polynomial degree/length.
        xi = np.linspace(-1, 1, 4)
        y = Tech.from_function(lambda x: x)
        p1 = ((y - float(xi[1])) * (y - float(xi[2])) * (y - float(xi[3]))) ** 2
        p = (y - float(xi[0])) * p1
        assert len(p) == 8 and len(p1) == 7
