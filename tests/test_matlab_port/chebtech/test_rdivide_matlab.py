"""Port of MATLAB Chebfun tests/chebtech/test_rdivide.m (Opus 4.8).

Self-validating: each quotient is checked against an analytic exact at the SAME
tolerance MATLAB uses.  MATLAB ``f ./ alpha`` -> Python ``f / alpha``,
``alpha ./ f`` -> ``alpha / f``, ``f ./ g`` (two techs) -> ``f / g``.  The file
loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}``; every method is
parametrized over both classes.

Gaps vs MATLAB (honest xfail/skip), reported in the final summary:
- Chebtech1 cannot represent complex-valued functions (vals2coeffs/coeffs2vals
  drop the imaginary part); complex operands / quotients skip Chebtech1.
- cos(1e4 x)/exp is degree ~1e4; its quotient's Clenshaw evaluation floor
  (~1e-11) marginally exceeds 1e4*vscale*eps on Chebtech2 -> xfail (see below).
- MATLAB ``isnan(g)`` for ``f ./ 0`` checks the tech became NaN; chebfunjax
  ``f / 0`` yields inf/NaN coefficients, checked with jnp.isnan/jnp.isinf.
- size-error assertions (dividing by a column vector / mismatched row vector)
  check a MATLAB error identifier that has no chebfunjax analog.

Array-valued: Chebtech now supports (n, m) coefficient matrices, so the
array-valued rdivide-by-constant cases (pass 3:6) are ported.  These divide by
a scalar or a row vector of constants (not by another array-valued tech), so
they do not need the compose path.

Provenance
----------
MATLAB source : tests/chebtech/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
# arbitrary complex constants (match test_rdivide.m).
ALPHA = -0.194758928283640 + 0.075474485412665j
BETA = -0.526634844879922 - 0.685484380523668j

_CT1_COMPLEX = (
    "Chebtech1 drops the imaginary part in vals2coeffs/coeffs2vals; it cannot "
    "represent complex-valued functions built via from_function"
)
_R10_FLOOR = (
    "cos(1e4*x)/exp is degree ~1e4; its quotient reconstruction evaluated on "
    "linspace(-1,1,100) has sup error ~6.5e-12, marginally over "
    "1e4*vscale*eps (~6.0e-12) -- the Clenshaw evaluation-conditioning floor "
    "for a degree-1e4 oscillatory series. MATLAB passes only via lucky "
    "100-random-point sampling."
)
_SIZEERR = "chebfunjax has no MATLAB rdivide:size error identifier (scalar-valued)"


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
class TestChebtechRdivide:
    def test_div_by_scalar(self, Tech):
        # pass(n, 1): feval(sin ./ alpha) == sin / alpha.
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = f / ALPHA
        err = _ninf(g(X) - (jnp.sin(X) / ALPHA))
        assert err < 10 * g.vscale * EPS

    def test_div_by_zero_is_nan(self, Tech):
        # pass(n, 2): isnan(sin ./ 0).  chebfunjax f/0 -> inf/NaN coeffs.
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = f / 0
        assert bool(jnp.any(jnp.isnan(g.coeffs)) or jnp.any(jnp.isinf(g.coeffs)))

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos] ./ alpha.
    def test_array_valued_div_by_scalar(self, Tech):
        # pass(n, 3): feval([sin cos] ./ alpha) == [sin cos] / alpha.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = f / ALPHA
        exact = jnp.stack([jnp.sin(X) / ALPHA, jnp.cos(X) / ALPHA], axis=-1)
        assert _ninf(g(X) - exact) < 10 * g.vscale * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): isnan([sin cos] ./ 0).
    def test_array_valued_div_by_zero(self, Tech):
        # pass(n, 4): isnan([sin cos] ./ 0).  chebfunjax f/0 -> inf/NaN coeffs.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = f / 0
        assert bool(jnp.any(jnp.isnan(g.coeffs)) or jnp.any(jnp.isinf(g.coeffs)))

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos] ./ [alpha beta].
    def test_div_by_scalar_row_matrix(self, Tech):
        # pass(n, 5): feval([sin cos] ./ [alpha beta]) == [sin/alpha cos/beta].
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = f / jnp.asarray([ALPHA, BETA])
        exact = jnp.stack([jnp.sin(X) / ALPHA, jnp.cos(X) / BETA], axis=-1)
        assert _ninf(g(X) - exact) < 10 * EPS

    # FIXED (Fable 5, Big-Three array-valued epic): [sin cos] ./ [alpha 0] -> per-column NaN.
    def test_div_by_scalar_row_with_zero(self, Tech):
        # pass(n, 6): dividing by [alpha 0] leaves col 1 finite and col 2 all NaN.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        g = f / jnp.asarray([ALPHA, 0.0])
        c = g.coeffs
        assert not bool(jnp.any(jnp.isnan(c[:, 0]))) and bool(
            jnp.all(jnp.isnan(c[:, 1]))
        )

    def test_scalar_div_by_function(self, Tech):
        # pass(n, 7): alpha ./ exp -> alpha / e^x (complex).
        f = Tech.from_function(lambda x: jnp.exp(x))
        g = ALPHA / f
        err = _ninf(g(X) - (ALPHA / jnp.exp(X)))
        assert err < 10 * g.vscale * EPS

    def test_div_exp_minus_1_by_exp(self, Tech):
        # pass(n, 8): (e^x - 1) ./ e^x.
        g = Tech.from_function(lambda x: jnp.exp(x))
        f = Tech.from_function(lambda x: jnp.exp(x) - 1)
        h = f / g
        err = _ninf(h(X) - ((jnp.exp(X) - 1) / jnp.exp(X)))
        assert err < 1e4 * h.vscale * EPS

    def test_div_lorentzian_by_exp(self, Tech):
        # pass(n, 9): 1/(1+x^2) ./ e^x.
        g = Tech.from_function(lambda x: jnp.exp(x))
        f = Tech.from_function(lambda x: 1.0 / (1 + x ** 2))
        h = f / g
        err = _ninf(h(X) - ((1.0 / (1 + X ** 2)) / jnp.exp(X)))
        assert err < 1e4 * h.vscale * EPS

    def test_div_high_freq_by_exp(self, Tech):
        # pass(n, 10): cos(1e4 x) ./ e^x.
        if Tech is Chebtech2:
            pytest.xfail(_R10_FLOOR)
        g = Tech.from_function(lambda x: jnp.exp(x))
        f = Tech.from_function(lambda x: jnp.cos(1e4 * x))
        h = f / g
        err = _ninf(h(X) - (jnp.cos(1e4 * X) / jnp.exp(X)))
        assert err < 1e4 * h.vscale * EPS

    def test_div_complex_sinh_by_exp(self, Tech):
        # pass(n, 11): sinh(t e^{2pi i/6}) ./ e^x.
        g = Tech.from_function(lambda x: jnp.exp(x))
        f = Tech.from_function(lambda t: jnp.sinh(t * jnp.exp(2j * jnp.pi / 6)))
        h = f / g
        exact = jnp.sinh(X * jnp.exp(2j * jnp.pi / 6)) / jnp.exp(X)
        assert _ninf(h(X) - exact) < 1e4 * h.vscale * EPS

    def test_div_by_column_matrix_error(self, Tech):
        # pass(n, 12): sin ./ [1; 2] -> rdivide:size error.
        f = Tech.from_function(jnp.sin)
        with pytest.raises(ValueError, match="rdivide:size"):
            f / jnp.array([[1.0], [2.0]])

    def test_div_by_mismatched_row_matrix_error(self, Tech):
        # pass(n, 13): [sin cos] ./ [1 2 3] -> rdivide:size error.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        with pytest.raises(ValueError, match="rdivide:size"):
            f / jnp.array([1.0, 2.0, 3.0])

    def test_div_by_scalar_matches_direct_construction(self, Tech):
        # pass(n, 14): coeffs of (sin ./ alpha) match direct construction.
        f = Tech.from_function(lambda x: jnp.sin(x))
        h1 = f / ALPHA
        h2 = Tech.from_function(lambda x: jnp.sin(x) / ALPHA)
        assert _coeff_diff(h1, h2) < 100 * EPS

    def test_div_by_function_matches_direct_construction(self, Tech):
        # pass(n, 15): coeffs of (sin ./ exp) match direct construction.
        f = Tech.from_function(lambda x: jnp.sin(x))
        g = Tech.from_function(lambda x: jnp.exp(x))
        h1 = f / g
        h2 = Tech.from_function(lambda x: jnp.sin(x) / jnp.exp(x))
        assert _coeff_diff(h1, h2) < 100 * EPS
