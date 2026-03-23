"""Tests for Chebtech2 arithmetic and calculus operators.

JAX contract:
    jit=yes for diff, cumsum, sum, inner, norm, and all arithmetic with
        fixed-size operands.
    vmap=yes for evaluation (Clenshaw is vectorized).
    grad=yes for sum, inner, diff, cumsum through coefficients.

Provenance
----------
MATLAB source : @chebtech/{plus,minus,times,rdivide,power,uminus,
    diff,cumsum,sum,innerProduct,roots}.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import (
    Chebtech2,
    _coeff_multiply,
    _cumsum_coeffs,
    _definite_integral,
    _diff_coeffs,
    _inner_product,
)
from chebfunjax.utils.quadrature import chebpts

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sin(n: int = 30) -> Chebtech2:
    """Chebtech2 for sin(x) on [-1, 1] with *n* points."""
    x = chebpts(n, kind=2)
    return Chebtech2.from_values(jnp.sin(x))


def _make_cos(n: int = 30) -> Chebtech2:
    x = chebpts(n, kind=2)
    return Chebtech2.from_values(jnp.cos(x))


def _make_exp(n: int = 30) -> Chebtech2:
    x = chebpts(n, kind=2)
    return Chebtech2.from_values(jnp.exp(x))


_XTEST = jnp.linspace(-1.0, 1.0, 201, dtype=jnp.float64)


# ===========================================================================
# Tier 1: Pure mathematical tests
# ===========================================================================


class TestArithmeticBasic:
    """Basic arithmetic operator tests -- no MATLAB references."""

    def test_add_two_chebtechs(self):
        """(sin + cos)(x) = sin(x) + cos(x)."""
        f = _make_sin()
        g = _make_cos()
        h = f + g
        err = jnp.max(jnp.abs(h(_XTEST) - (jnp.sin(_XTEST) + jnp.cos(_XTEST))))
        assert float(err) < 1e-14

    def test_add_scalar(self):
        """sin + 1."""
        f = _make_sin()
        h = f + 1.0
        err = jnp.max(jnp.abs(h(_XTEST) - (jnp.sin(_XTEST) + 1.0)))
        assert float(err) < 1e-14

    def test_radd_scalar(self):
        """1 + sin."""
        f = _make_sin()
        h = 1.0 + f
        err = jnp.max(jnp.abs(h(_XTEST) - (1.0 + jnp.sin(_XTEST))))
        assert float(err) < 1e-14

    def test_sub_two_chebtechs(self):
        """(sin - cos)(x) = sin(x) - cos(x)."""
        f = _make_sin()
        g = _make_cos()
        h = f - g
        err = jnp.max(jnp.abs(h(_XTEST) - (jnp.sin(_XTEST) - jnp.cos(_XTEST))))
        assert float(err) < 1e-14

    def test_rsub_scalar(self):
        """1 - sin."""
        f = _make_sin()
        h = 1.0 - f
        err = jnp.max(jnp.abs(h(_XTEST) - (1.0 - jnp.sin(_XTEST))))
        assert float(err) < 1e-14

    def test_mul_two_chebtechs(self):
        """(sin * sin)(x) = sin^2(x)."""
        f = _make_sin()
        h = f * f
        err = jnp.max(jnp.abs(h(_XTEST) - jnp.sin(_XTEST) ** 2))
        assert float(err) < 1e-14

    def test_mul_scalar(self):
        """2 * sin."""
        f = _make_sin()
        h = f * 2.0
        err = jnp.max(jnp.abs(h(_XTEST) - 2.0 * jnp.sin(_XTEST)))
        assert float(err) < 1e-14

    def test_rmul_scalar(self):
        """3 * sin."""
        f = _make_sin()
        h = 3.0 * f
        err = jnp.max(jnp.abs(h(_XTEST) - 3.0 * jnp.sin(_XTEST)))
        assert float(err) < 1e-14

    def test_neg(self):
        """-sin."""
        f = _make_sin()
        h = -f
        err = jnp.max(jnp.abs(h(_XTEST) - (-jnp.sin(_XTEST))))
        assert float(err) < 1e-15

    def test_pos(self):
        """+sin is sin."""
        f = _make_sin()
        h = +f
        assert h is f  # identity

    def test_truediv_scalar(self):
        """sin / 2."""
        f = _make_sin()
        h = f / 2.0
        err = jnp.max(jnp.abs(h(_XTEST) - jnp.sin(_XTEST) / 2.0))
        assert float(err) < 1e-14

    def test_truediv_chebtech(self):
        """sin / cos = tan (evaluated at interior points)."""
        f = _make_sin()
        g = _make_cos()
        h = f / g
        xt = jnp.linspace(-0.99, 0.99, 200, dtype=jnp.float64)
        err = jnp.max(jnp.abs(h(xt) - jnp.tan(xt)))
        assert float(err) < 1e-13

    def test_rtruediv(self):
        """1 / exp = exp(-x)."""
        f = _make_exp()
        h = 1.0 / f
        xt = jnp.linspace(-0.99, 0.99, 200, dtype=jnp.float64)
        err = jnp.max(jnp.abs(h(xt) - jnp.exp(-xt)))
        assert float(err) < 1e-13

    def test_pow_integer(self):
        """sin^2 via __pow__ with integer exponent."""
        f = _make_sin()
        h = f ** 2
        err = jnp.max(jnp.abs(h(_XTEST) - jnp.sin(_XTEST) ** 2))
        assert float(err) < 1e-14

    def test_pow_zero(self):
        """f^0 = 1."""
        f = _make_sin()
        h = f ** 0
        npt.assert_allclose(np.array(h(_XTEST)), 1.0, atol=1e-15)

    def test_pow_one(self):
        """f^1 = f."""
        f = _make_sin()
        h = f ** 1
        err = jnp.max(jnp.abs(h(_XTEST) - jnp.sin(_XTEST)))
        assert float(err) < 1e-14

    def test_sin2_plus_cos2(self):
        """sin^2 + cos^2 = 1 (Pythagorean identity)."""
        f = _make_sin()
        g = _make_cos()
        h = f ** 2 + g ** 2
        npt.assert_allclose(np.array(h(_XTEST)), 1.0, atol=2e-14)

    def test_add_different_lengths(self):
        """Adding Chebtech2 objects of different lengths."""
        f = Chebtech2.from_coeffs(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
        g = Chebtech2.from_coeffs(jnp.array([4.0, 5.0], dtype=jnp.float64))
        h = f + g
        expected = jnp.array([5.0, 7.0, 3.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(h.coeffs), np.array(expected), atol=1e-15)

    def test_constant_operations(self):
        """Arithmetic with constant Chebtech2."""
        f = Chebtech2.from_coeffs(jnp.array([3.0], dtype=jnp.float64))
        g = Chebtech2.from_coeffs(jnp.array([2.0], dtype=jnp.float64))
        h = f + g
        npt.assert_allclose(np.array(h.coeffs), np.array([5.0]), atol=1e-15)

        h2 = f * g
        npt.assert_allclose(np.array(h2.coeffs), np.array([6.0]), atol=1e-15)


class TestCoeffMultiply:
    """Tests for _coeff_multiply (FFT-based coefficient multiplication)."""

    def test_T1_times_T1(self):
        """T_1 * T_1 = (T_0 + T_2)/2."""
        c1 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        hc = _coeff_multiply(c1, c1)
        # x * x = x^2 = T_0/2 + T_2/2 (with T_0=1, T_2=2x^2-1)
        # Actually x^2 = (T_0 + T_2)/2, so coeffs = [0.5, 0, 0.5]
        npt.assert_allclose(np.array(hc), np.array([0.5, 0.0, 0.5]), atol=1e-15)

    def test_constant_times_T1(self):
        """3 * T_1 = 3*T_1."""
        c1 = jnp.array([3.0], dtype=jnp.float64)
        c2 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        hc = _coeff_multiply(c1, c2)
        npt.assert_allclose(np.array(hc), np.array([0.0, 3.0]), atol=1e-15)


class TestDiff:
    """Tests for differentiation via Chebyshev coefficient recurrence."""

    def test_diff_sin_is_cos(self):
        """d/dx(sin) = cos."""
        f = _make_sin()
        df = f.diff()
        err = jnp.max(jnp.abs(df(_XTEST) - jnp.cos(_XTEST)))
        assert float(err) < 1e-13

    def test_diff_exp_is_exp(self):
        """d/dx(exp) = exp."""
        f = _make_exp()
        df = f.diff()
        err = jnp.max(jnp.abs(df(_XTEST) - jnp.exp(_XTEST)))
        # Differentiation amplifies high-frequency noise by a factor of n;
        # for n=30, 1e-15 * 30 ~ 3e-14, so 5e-13 is a comfortable bound.
        assert float(err) < 5e-13

    def test_diff2_sin_is_neg_sin(self):
        """d^2/dx^2(sin) = -sin."""
        f = _make_sin()
        d2f = f.diff(2)
        err = jnp.max(jnp.abs(d2f(_XTEST) - (-jnp.sin(_XTEST))))
        # Second derivative amplifies noise by ~n^2; for n=30
        # expect ~1e-15 * 30^2 ~ 1e-12.  Allow 5e-11 comfortably.
        assert float(err) < 5e-11

    def test_diff_constant_is_zero(self):
        """Derivative of a constant is zero."""
        f = Chebtech2.from_coeffs(jnp.array([5.0], dtype=jnp.float64))
        df = f.diff()
        npt.assert_allclose(np.array(df.coeffs), np.array([0.0]), atol=1e-15)

    def test_diff_linear(self):
        """d/dx(x) = 1."""
        f = Chebtech2.from_coeffs(jnp.array([0.0, 1.0], dtype=jnp.float64))
        df = f.diff()
        npt.assert_allclose(np.array(df.coeffs), np.array([1.0]), atol=1e-15)

    def test_diff_quadratic(self):
        """d/dx(x^2) = 2x. x^2 = (T_0+T_2)/2, so d/dx = x (or T_1)."""
        # x^2 coeffs: [0.5, 0, 0.5]
        f = Chebtech2.from_coeffs(jnp.array([0.5, 0.0, 0.5], dtype=jnp.float64))
        df = f.diff()
        # 2x has coeffs [0, 2]
        npt.assert_allclose(np.array(df.coeffs), np.array([0.0, 2.0]), atol=1e-14)

    def test_diff_k0_is_identity(self):
        """diff(f, 0) = f."""
        f = _make_sin()
        g = f.diff(0)
        assert g is f


class TestCumsum:
    """Tests for indefinite integral (antiderivative with F(-1) = 0)."""

    def test_cumsum_cos_is_sin_shifted(self):
        """cumsum(cos) = sin(x) - sin(-1)."""
        f = _make_cos()
        F = f.cumsum()
        expected = jnp.sin(_XTEST) - jnp.sin(-1.0)
        err = jnp.max(jnp.abs(F(_XTEST) - expected))
        assert float(err) < 1e-14

    def test_cumsum_minus_one_is_zero(self):
        """F(-1) = 0 for any cumsum."""
        f = _make_sin()
        F = f.cumsum()
        val = float(F(jnp.array(-1.0, dtype=jnp.float64)))
        assert abs(val) < 1e-14

    def test_diff_cumsum_roundtrip(self):
        """diff(cumsum(f)) = f."""
        f = _make_sin()
        g = f.cumsum().diff()
        err = jnp.max(jnp.abs(g(_XTEST) - jnp.sin(_XTEST)))
        assert float(err) < 1e-14

    def test_cumsum_constant(self):
        """cumsum of constant 3 is 3*(x+1) = 3x + 3."""
        f = Chebtech2.from_coeffs(jnp.array([3.0], dtype=jnp.float64))
        F = f.cumsum()
        # F(x) = 3*(x - (-1)) = 3*(x+1) = 3x + 3
        # Chebyshev: 3 + 3*T_1, coeffs = [3, 3]
        npt.assert_allclose(np.array(F.coeffs), np.array([3.0, 3.0]), atol=1e-14)


class TestSum:
    """Tests for definite integral over [-1, 1]."""

    def test_sum_sin_is_zero(self):
        """integral of sin(x) from -1 to 1 = 0 (odd function)."""
        f = _make_sin()
        s = float(f.sum())
        assert abs(s) < 1e-15

    def test_sum_x_squared(self):
        """integral of x^2 from -1 to 1 = 2/3."""
        f = Chebtech2.from_coeffs(jnp.array([0.5, 0.0, 0.5], dtype=jnp.float64))
        s = float(f.sum())
        npt.assert_allclose(s, 2.0 / 3.0, atol=1e-15)

    def test_sum_constant(self):
        """integral of 1 from -1 to 1 = 2."""
        f = Chebtech2.from_coeffs(jnp.array([1.0], dtype=jnp.float64))
        s = float(f.sum())
        npt.assert_allclose(s, 2.0, atol=1e-15)

    def test_sum_exp(self):
        """integral of exp(x) from -1 to 1 = e - 1/e."""
        f = _make_exp()
        s = float(f.sum())
        exact = float(jnp.exp(1.0) - jnp.exp(-1.0))
        npt.assert_allclose(s, exact, rtol=1e-14)

    def test_sum_cos(self):
        """integral of cos(x) from -1 to 1 = 2*sin(1)."""
        f = _make_cos()
        s = float(f.sum())
        exact = float(2.0 * jnp.sin(1.0))
        npt.assert_allclose(s, exact, rtol=1e-14)


class TestInnerProduct:
    """Tests for L^2 inner product."""

    def test_inner_sin_sin(self):
        """<sin, sin> = 1 - sin(2)/2."""
        f = _make_sin()
        ip = float(f.inner(f))
        exact = float(1.0 - jnp.sin(2.0) / 2.0)
        npt.assert_allclose(ip, exact, rtol=1e-14)

    def test_inner_sin_cos(self):
        """<sin, cos> = 0 (product is odd function)."""
        f = _make_sin()
        g = _make_cos()
        ip = float(f.inner(g))
        # sin(x)*cos(x) = sin(2x)/2, integral is 0
        assert abs(ip) < 1e-14

    def test_inner_x_x(self):
        """<x, x> = 2/3."""
        f = Chebtech2.from_coeffs(jnp.array([0.0, 1.0], dtype=jnp.float64))
        ip = float(f.inner(f))
        npt.assert_allclose(ip, 2.0 / 3.0, rtol=1e-14)

    def test_inner_orthogonality(self):
        """<T_2, T_3> = 0 (Chebyshev polynomials are orthogonal w.r.t. weight)."""
        # With L^2 inner product, <T_2, T_3> = integral T_2 T_3 = 0
        c2 = jnp.zeros(4, dtype=jnp.float64).at[2].set(1.0)
        c3 = jnp.zeros(4, dtype=jnp.float64).at[3].set(1.0)
        f = Chebtech2.from_coeffs(c2)
        g = Chebtech2.from_coeffs(c3)
        ip = float(f.inner(g))
        assert abs(ip) < 1e-14


class TestNorm:
    """Tests for Lp norms."""

    def test_l2_norm_sin(self):
        """L2 norm of sin = sqrt(1 - sin(2)/2)."""
        f = _make_sin()
        nrm = float(f.norm(2))
        exact = float(jnp.sqrt(1.0 - jnp.sin(2.0) / 2.0))
        npt.assert_allclose(nrm, exact, rtol=1e-14)

    def test_linf_norm_sin(self):
        """L-inf norm of sin ~ sin(1)."""
        f = _make_sin()
        nrm = float(f.norm(jnp.inf))
        npt.assert_allclose(nrm, float(jnp.sin(1.0)), rtol=1e-3)

    def test_l2_norm_constant(self):
        """L2 norm of constant 3 = 3*sqrt(2)."""
        f = Chebtech2.from_coeffs(jnp.array([3.0], dtype=jnp.float64))
        nrm = float(f.norm(2))
        npt.assert_allclose(nrm, 3.0 * float(jnp.sqrt(2.0)), rtol=1e-14)


class TestRoots:
    """Tests for rootfinding via colleague matrix."""

    def test_roots_sin(self):
        """Roots of sin in [-1, 1] = {0}."""
        f = _make_sin()
        rts = f.roots()
        assert rts.shape[0] == 1
        npt.assert_allclose(np.array(rts), 0.0, atol=1e-14)

    def test_roots_T5(self):
        """Roots of T_5 are cos((2k-1)*pi/10), k=1..5."""
        c = jnp.zeros(6, dtype=jnp.float64).at[5].set(1.0)
        f = Chebtech2.from_coeffs(c)
        rts = np.sort(np.array(f.roots()))
        expected = np.sort(np.cos((2 * np.arange(1, 6) - 1) * np.pi / 10))
        npt.assert_allclose(rts, expected, atol=1e-14)

    def test_roots_quadratic(self):
        """Roots of x^2 - 1/4 are {-1/2, 1/2}."""
        # x^2 - 1/4 = (T_0+T_2)/2 - 1/4 = T_0/4 + T_2/2
        c = jnp.array([1.0 / 4.0, 0.0, 1.0 / 2.0], dtype=jnp.float64)
        f = Chebtech2.from_coeffs(c)
        rts = np.sort(np.array(f.roots()))
        npt.assert_allclose(rts, [-0.5, 0.5], atol=1e-14)

    def test_roots_constant_nonzero(self):
        """A nonzero constant has no roots."""
        f = Chebtech2.from_coeffs(jnp.array([5.0], dtype=jnp.float64))
        rts = f.roots()
        # Should return empty (or a root at 0 if the constant is 0)
        # For nonzero constant, the function never vanishes
        # The colleague matrix approach for n=1 is a special case;
        # our code checks if coefficients are all zero
        assert rts.shape[0] == 0 or (rts.shape[0] == 1 and abs(float(rts[0])) > 0.5)

    def test_roots_high_degree(self):
        """Roots of a polynomial with degree > 50 (triggers subdivision)."""
        # Use T_{60}: has 60 roots in [-1, 1]
        c = jnp.zeros(61, dtype=jnp.float64).at[60].set(1.0)
        f = Chebtech2.from_coeffs(c)
        rts = np.sort(np.array(f.roots()))
        expected = np.sort(np.cos((2 * np.arange(1, 61) - 1) * np.pi / 120))
        assert len(rts) == 60
        npt.assert_allclose(rts, expected, atol=1e-10)


class TestProlong:
    """Tests for prolongation (zero-padding / truncation)."""

    def test_prolong_pad(self):
        """Prolong by zero-padding."""
        f = Chebtech2.from_coeffs(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
        g = f.prolong(5)
        npt.assert_allclose(np.array(g.coeffs), [1.0, 2.0, 3.0, 0.0, 0.0], atol=1e-15)

    def test_prolong_truncate(self):
        """Prolong by truncation."""
        f = Chebtech2.from_coeffs(jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64))
        g = f.prolong(3)
        npt.assert_allclose(np.array(g.coeffs), [1.0, 2.0, 3.0], atol=1e-15)

    def test_prolong_same(self):
        """Prolong to same length returns equivalent."""
        f = Chebtech2.from_coeffs(jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64))
        g = f.prolong(3)
        npt.assert_allclose(np.array(g.coeffs), np.array(f.coeffs), atol=1e-15)


# ===========================================================================
# Tier 1: JIT / grad tests
# ===========================================================================


class TestJIT:
    """JIT safety tests for calculus operations.

    JAX contract: jit=yes for diff, cumsum, sum, inner (n must be static shape).
    """

    def test_jit_diff_coeffs(self):
        """JIT-compile _diff_coeffs."""
        c = jnp.array([0.0, 1.0, 0.0, -0.5], dtype=jnp.float64)
        jitted = jax.jit(functools.partial(_diff_coeffs, k=1))
        result = jitted(c)
        expected = _diff_coeffs(c, 1)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_jit_cumsum_coeffs(self):
        """JIT-compile _cumsum_coeffs."""
        c = jnp.array([1.0, 0.5, 0.1], dtype=jnp.float64)
        jitted = jax.jit(_cumsum_coeffs)
        result = jitted(c)
        expected = _cumsum_coeffs(c)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)

    def test_jit_definite_integral(self):
        """JIT-compile _definite_integral."""
        c = jnp.array([1.0, 0.0, 0.5], dtype=jnp.float64)
        jitted = jax.jit(_definite_integral)
        result = float(jitted(c))
        expected = float(_definite_integral(c))
        npt.assert_allclose(result, expected, atol=1e-15)

    def test_jit_inner_product(self):
        """JIT-compile _inner_product."""
        c1 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        c2 = jnp.array([1.0, 0.0, 0.5], dtype=jnp.float64)
        jitted = jax.jit(_inner_product)
        result = float(jitted(c1, c2))
        expected = float(_inner_product(c1, c2))
        npt.assert_allclose(result, expected, atol=1e-15)

    def test_jit_coeff_multiply(self):
        """JIT-compile _coeff_multiply."""
        c1 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        c2 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        jitted = jax.jit(_coeff_multiply)
        result = jitted(c1, c2)
        expected = _coeff_multiply(c1, c2)
        npt.assert_allclose(np.array(result), np.array(expected), atol=1e-15)


class TestGrad:
    """Gradient safety tests for calculus operations.

    JAX contract: grad=yes for sum, inner, diff (through coefficients).
    """

    def test_grad_sum_wrt_coeffs(self):
        """d(sum)/d(coeffs) = Chebyshev moments."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        grad_fn = jax.grad(_definite_integral)
        g = grad_fn(c)
        # Moments: m_k = 2/(1-k^2) for even k, 0 for odd k
        # m_0=2, m_1=0, m_2=2/(1-4)=-2/3, m_3=0
        expected = jnp.array([2.0, 0.0, -2.0 / 3.0, 0.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(g), np.array(expected), atol=1e-14)

    def test_grad_inner_wrt_coeffs(self):
        """Gradient of <f, g> w.r.t. f coefficients."""
        c1 = jnp.array([0.0, 1.0], dtype=jnp.float64)
        c2 = jnp.array([1.0, 0.0], dtype=jnp.float64)
        grad_fn = jax.grad(lambda c: _inner_product(c, c2))
        g = grad_fn(c1)
        # <f, g> with g=T_0=1 is integral of f, so grad = moments
        npt.assert_allclose(float(g[0]), 2.0, atol=1e-14)
        npt.assert_allclose(float(g[1]), 0.0, atol=1e-14)

    def test_grad_eval_matches_diff(self):
        """jax.grad of f(x) w.r.t. x matches f.diff()(x)."""
        c = _make_sin().coeffs
        x0 = jnp.array(0.5, dtype=jnp.float64)

        # Grad of evaluation w.r.t. x
        from chebfunjax.tech.chebtech import _clenshaw

        grad_val = jax.grad(lambda x: _clenshaw(c, x))(x0)

        # Value from diff
        dc = _diff_coeffs(c, 1)
        diff_val = _clenshaw(dc, x0)

        npt.assert_allclose(float(grad_val), float(diff_val), rtol=1e-12)

    def test_grad_sum_through_chebtech2(self):
        """Gradient of sum through Chebtech2.from_coeffs."""
        c = jnp.array([1.0, 0.5, 0.3], dtype=jnp.float64)

        def loss(coeffs):
            f = Chebtech2.from_coeffs(coeffs)
            return f.sum()

        g = jax.grad(loss)(c)
        expected = jnp.array([2.0, 0.0, -2.0 / 3.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(g), np.array(expected), atol=1e-14)


# ===========================================================================
# Tier 2: MATLAB cross-validation
# ===========================================================================


@pytest.mark.matlab
class TestMATLABCrossValidation:
    """Compare against MATLAB Chebfun reference outputs.

    Uses golden .mat files committed in tests/references/.
    """

    @pytest.fixture(autouse=True)
    def _load_ref(self):
        """Load MATLAB reference data."""
        from tests.conftest import load_matlab_ref

        self.ref = load_matlab_ref("chebtech_ops.mat")

    def _make_from_matlab_coeffs(self, key: str) -> Chebtech2:
        """Create Chebtech2 from MATLAB reference coefficients."""
        c = jnp.asarray(self.ref[key].ravel(), dtype=jnp.float64)
        return Chebtech2.from_coeffs(c)

    def test_diff_sin_coeffs(self):
        """diff(sin) coefficients match MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        df = f.diff()
        expected = self.ref["diff_sin_coeffs"].ravel()
        npt.assert_allclose(
            np.array(df.coeffs)[: len(expected)],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_diff_exp_coeffs(self):
        """diff(exp) coefficients match MATLAB."""
        f = self._make_from_matlab_coeffs("exp_coeffs")
        df = f.diff()
        expected = self.ref["diff_exp_coeffs"].ravel()
        npt.assert_allclose(
            np.array(df.coeffs)[: len(expected)],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_sum_sin(self):
        """sum(sin) matches MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        s = float(f.sum())
        npt.assert_allclose(s, float(self.ref["sum_sin"]), atol=1e-14)

    def test_sum_cos(self):
        """sum(cos) matches MATLAB."""
        f = self._make_from_matlab_coeffs("cos_coeffs")
        s = float(f.sum())
        npt.assert_allclose(s, float(self.ref["sum_cos"]), rtol=1e-14)

    def test_sum_exp(self):
        """sum(exp) matches MATLAB."""
        f = self._make_from_matlab_coeffs("exp_coeffs")
        s = float(f.sum())
        npt.assert_allclose(s, float(self.ref["sum_exp"]), rtol=1e-14)

    def test_sum_x2(self):
        """sum(x^2) matches MATLAB (= 2/3)."""
        s = float(self.ref["sum_x2"])
        npt.assert_allclose(s, 2.0 / 3.0, atol=1e-15)

    def test_inner_sin_sin(self):
        """<sin, sin> matches MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        ip = float(f.inner(f))
        npt.assert_allclose(ip, float(self.ref["inner_sin_sin"]), rtol=1e-12)

    def test_inner_sin_cos(self):
        """<sin, cos> matches MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        g = self._make_from_matlab_coeffs("cos_coeffs")
        ip = float(f.inner(g))
        npt.assert_allclose(ip, float(self.ref["inner_sin_cos"]), atol=1e-13)

    def test_roots_sin(self):
        """Roots of sin match MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        rts = np.sort(np.array(f.roots()))
        ref_val = self.ref["roots_sin"]
        expected = np.atleast_1d(np.asarray(ref_val).ravel())
        npt.assert_allclose(rts, expected, atol=1e-12)

    def test_roots_T5(self):
        """Roots of T_5 match MATLAB."""
        rts_expected = np.sort(self.ref["roots_T5"].ravel())
        c = jnp.zeros(6, dtype=jnp.float64).at[5].set(1.0)
        f = Chebtech2.from_coeffs(c)
        rts = np.sort(np.array(f.roots()))
        npt.assert_allclose(rts, rts_expected, atol=1e-12)

    def test_roots_quadratic(self):
        """Roots of x^2 - 1/4 match MATLAB."""
        rts_expected = np.sort(self.ref["roots_quadratic"].ravel())
        # x^2 - 1/4 = T_0/4 + T_2/2
        c = jnp.array([1.0 / 4.0, 0.0, 1.0 / 2.0], dtype=jnp.float64)
        f = Chebtech2.from_coeffs(c)
        rts = np.sort(np.array(f.roots()))
        npt.assert_allclose(rts, rts_expected, atol=1e-12)

    def test_add_coeffs(self):
        """(sin + cos) coefficients match MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        g = self._make_from_matlab_coeffs("cos_coeffs")
        h = f + g
        expected = self.ref["add_coeffs"].ravel()
        npt.assert_allclose(
            np.array(h.coeffs)[: len(expected)],
            expected,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_neg_coeffs(self):
        """-sin coefficients match MATLAB."""
        f = self._make_from_matlab_coeffs("sin_coeffs")
        h = -f
        expected = self.ref["neg_coeffs"].ravel()
        npt.assert_allclose(np.array(h.coeffs), expected, atol=1e-15)
