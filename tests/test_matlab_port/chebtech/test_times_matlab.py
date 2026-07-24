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
- chebfunjax arithmetic (from_coeffs) always sets ishappy=True; happiness is
  not propagated through times (the "unhappy result" checks xfail).
- empty / dimension-error assertions have no scalar-tech analog.

Array-valued: Chebtech now supports (n, m) coefficient matrices and has a
``conj`` method, so the array-valued times cases (pass 4:5, 7, 12:14, 22) and
the ``f .* conj(f)`` cases (pass 17:18) are ported.  Multiplication by a
complex scalar promotes the coeff dtype on both classes, so the array-valued
exact check (pass 5) holds on Chebtech1 too.

Provenance
----------
MATLAB source : tests/chebtech/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
# arbitrary complex multiplicative constants (match test_times.m).
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j

_CT1_COMPLEX = (
    "Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it cannot "
    "represent complex-valued functions built via from_function"
)
_UNHAPPY = (
    "chebfunjax Chebtech arithmetic builds results via from_coeffs, which "
    "always sets ishappy=True; happiness is not propagated through times"
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
class TestChebtechTimes:
    def test_empty_arguments(self, Tech):
        # pass(n, 1): isempty(f .* f) && isempty(f .* g) && isempty(g .* f)
        f = Tech.empty()
        g = Tech.from_function(lambda x: x)
        assert (f * f).isempty()
        assert (f * g).isempty()
        assert (g * f).isempty()

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

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos] .* alpha.
    def test_array_valued_mult_scalar_commutes(self, Tech):
        # pass(n, 4): isequal([sin cos] .* alpha, alpha .* [sin cos]).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        assert _coeff_diff(f * ALPHA, ALPHA * f) == 0.0

    def test_array_valued_mult_scalar_matches_exact(self, Tech):
        # pass(n, 5): feval([sin cos] .* alpha) == [sin cos] * alpha.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g1 = f * ALPHA
        exact = jnp.stack([jnp.sin(X), jnp.cos(X)], axis=-1) * ALPHA
        assert _ninf(g1(X) - exact) < 10 * g1.vscale * EPS

    # -- Multiplication by constant functions. pass(n, 6). --
    def test_mult_by_constant_function(self, Tech):
        # pass(n, 6): sin .* (alpha * ones).
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = Tech.from_function(lambda x: ALPHA * jnp.ones_like(x))
        h = f * g
        err = _ninf(h(X) - (jnp.sin(X) * ALPHA))
        assert err < 1e4 * h.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos] .* [alpha beta] (constant columns).
    def test_array_valued_mult_constants(self, Tech):
        # pass(n, 7): [sin cos] .* [alpha beta] (constant complex columns).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack(
                [ALPHA * jnp.ones_like(x), BETA * jnp.ones_like(x)], axis=-1
            )
        )
        h = f * g
        exact = jnp.stack([jnp.sin(X) * ALPHA, jnp.cos(X) * BETA], axis=-1)
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

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
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        g = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f * g
        exact = (jnp.exp(X) - 1) * jnp.sinh(X * jnp.exp(2j * jnp.pi / 6))
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos exp] .* tanh.
    def test_array_valued_mult_scalar_tech_commutes(self, Tech):
        # pass(n, 12): norm(coeffs(f .* g) - coeffs(g .* f)) < 10*eps, g scalar tanh.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(lambda x: jnp.tanh(x))
        assert _coeff_diff(f * g, g * f) < 10 * EPS

    def test_array_valued_mult_scalar_tech_exact(self, Tech):
        # pass(n, 13): feval([sin cos exp] .* tanh) == exact, tol 10*eps.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(lambda x: jnp.tanh(x))
        h = f * g
        exact = jnp.stack(
            [jnp.tanh(X) * jnp.sin(X), jnp.tanh(X) * jnp.cos(X), jnp.tanh(X) * jnp.exp(X)],
            axis=-1,
        )
        assert _ninf(h(X) - exact) < 10 * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos exp] .* [sinh cosh tanh].
    def test_array_valued_mult_array(self, Tech):
        # pass(n, 14): feval([sin cos exp] .* [sinh cosh tanh]) == exact, tol 10*eps.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sinh(x), jnp.cosh(x), jnp.tanh(x)], axis=-1)
        )
        h = f * g
        exact = jnp.stack(
            [jnp.sinh(X) * jnp.sin(X), jnp.cosh(X) * jnp.cos(X), jnp.tanh(X) * jnp.exp(X)],
            axis=-1,
        )
        assert _ninf(h(X) - exact) < 10 * EPS

    def test_array_valued_dim_mismatch(self, Tech):
        # pass(n, 15): array-valued .* with mismatched columns -> dim2 error.
        pytest.skip(_DIMERR)

    # -- Specially handled cases (positivity adjustments). pass(n, 16:20). --
    def test_mult_complex_sinh_squared(self, Tech):
        # pass(n, 16): sinh(t e^{2pi i/6}) .* itself.
        f = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f * f
        exact = jnp.sinh(X * jnp.exp(2j * jnp.pi / 6)) ** 2
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): f .* conj(f).
    def test_mult_by_conjugate_accuracy(self, Tech):
        # pass(n, 17): feval(f .* conj(f)) == sinh(t w) * conj(sinh(t w)).
        f = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        g = f.conj()
        h = f * g
        sinh_val = jnp.sinh(X * jnp.exp(2j * jnp.pi / 6))
        exact = sinh_val * jnp.conj(sinh_val)
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    def test_mult_by_conjugate_positivity(self, Tech):
        # pass(n, 18): f .* conj(f) is nonnegative on the Chebyshev grid.
        f = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        g = f.conj()
        h = f * g
        tol = 1e4 * h.vscale * EPS
        values = jnp.real(Tech.coeffs2vals(h.coeffs))
        assert bool(jnp.all(values >= -tol))

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

    # FIXED (Fable 5, Big-Three array-valued epic): [f...] .* [f...] moderate dims.
    def test_array_valued_moderate_dims(self, Tech):
        # pass(n, 22): squaring an 8-col vs 20-col array-valued tech agrees in col 1.
        # MATLAB builds f8 = [f f f f f f f f], f20 = [f8 f8 f f f f]; the first
        # column of f.^2 must not depend on how many columns are concatenated.
        f = Tech.from_function(lambda x: x)
        f8 = Tech.cell2mat([f] * 8)
        f20 = Tech.cell2mat([f] * 20)
        f8_2 = f8 * f8
        f20_2 = f20 * f20
        assert _ninf(f8_2.coeffs[:, 0] - f20_2.coeffs[:, 0]) < 50 * EPS

    def test_unhappy_times_happy_stays_unhappy(self, Tech):
        # pass(n, 23): f (happy) .* g (unhappy) -> unhappy.
        # FIXED: arithmetic propagates ishappy (self.ishappy and other.ishappy).
        f = Tech.from_function(lambda x: jnp.cos(x + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Tech.from_function(lambda x: jnp.sqrt(x + 1))  # unhappy
        h = f * g
        assert (not g.ishappy) and (not h.ishappy)

    def test_happy_times_unhappy_stays_unhappy(self, Tech):
        # pass(n, 24): g (unhappy) .* f (happy) -> unhappy.
        f = Tech.from_function(lambda x: jnp.cos(x + 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            g = Tech.from_function(lambda x: jnp.sqrt(x + 1))  # unhappy
        h = g * f
        assert (not g.ishappy) and (not h.ishappy)

    def test_polynomial_product_degree(self, Tech):
        # pass(n, 25): products give the correct polynomial degree/length.
        xi = np.linspace(-1, 1, 4)
        y = Tech.from_function(lambda x: x)
        p1 = ((y - float(xi[1])) * (y - float(xi[2])) * (y - float(xi[3]))) ** 2
        p = (y - float(xi[0])) * p1
        assert len(p) == 8 and len(p1) == 7
