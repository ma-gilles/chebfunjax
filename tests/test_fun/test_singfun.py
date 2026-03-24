"""Tests for chebfunjax.fun.singfun — Singfun class.

Covers algebraic singularities at endpoints via f(x) = s(x)*(1+x)^a*(1-x)^b.

JAX contract:
- __call__: jit=YES, vmap=YES, grad=YES (w.r.t. evaluation point)
- sum: jit=YES (exponents are static)
- diff: construction NOT jit-safe; result evaluation IS jit-safe
- Arithmetic: construction NOT jit-safe (may call from_function adaptively)

Test cases:
- sqrt(1-x^2) = (1+x)^0.5 (1-x)^0.5 * 1  — integral = pi/2
- (1+x)^(-0.5)  — integrable singularity at left endpoint, integral = 2*sqrt(2)
- (1-x^2)^(-0.5) = (1+x)^(-0.5)*(1-x)^(-0.5)*1  — integral = pi
- Trivial smooth: exponents=(0,0)  — delegates to Chebtech2
- Divergent integrals (exponent <= -1)  — returns ±inf or nan

Notes on construction API:
  - Singfun.from_function(f, (a, b)) extracts the smooth factor
      s(x) = f(x) / (1+x)^a / (1-x)^b
    and approximates it.  The resulting Singfun represents f(x) = s(x)*weight.
  - Singfun.from_chebtech(tech, (a, b)) directly uses tech as the smooth factor.
  - To integrate the Jacobi weight (1+x)^a*(1-x)^b itself, use from_chebtech
    with a constant-1 Chebtech2.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.fun.singfun import Singfun
from chebfunjax.tech.chebtech import Chebtech2

RTOL = 1e-12
ATOL = 1e-13


# =============================================================================
# Tier 1: Construction
# =============================================================================


class TestSingfunConstruction:
    """Tests for Singfun construction.

    JAX contract: construction NOT jit-safe.
    """

    def test_from_function_smooth(self):
        """Smooth function with zero exponents wraps a Chebtech2 of degree ~1."""
        sf = Singfun.from_function(lambda x: jnp.ones_like(x, dtype=jnp.float64), (0.0, 0.0))
        assert sf.issmooth
        assert sf.n >= 1

    def test_from_function_sqrt_1mx2(self):
        """sqrt(1-x^2) resolved with exponents (0.5, 0.5); smooth factor is 1."""
        sf = Singfun.from_function(lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5))
        assert sf.issmooth is False  # exponents are non-zero
        # Smooth factor = sqrt(1-x^2)/(1+x)^0.5/(1-x)^0.5 = 1.0 (constant)
        assert sf.n <= 4

    def test_from_function_left_sing(self):
        """(1+x)^(-0.5) with exponent (-0.5, 0.0): smooth factor is 1."""
        sf = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        assert sf.n <= 4

    def test_from_chebtech(self):
        """from_chebtech wraps an existing Chebtech2 with given exponents."""
        tech = Chebtech2.from_function(lambda x: jnp.ones_like(x, dtype=jnp.float64))
        sf = Singfun.from_chebtech(tech, (0.5, 0.5))
        assert sf.exponents == (0.5, 0.5)
        assert sf.smoothPart is tech

    def test_from_function_fixed_n(self):
        """Fixed n construction."""
        sf = Singfun.from_function(lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5), n=8)
        assert sf.n == 8

    def test_exponents_stored_as_floats(self):
        """Exponents are stored as (float, float) tuple."""
        sf = Singfun.from_chebtech(
            Chebtech2.from_function(lambda x: jnp.ones_like(x, dtype=jnp.float64)),
            (1, -1),
        )
        assert sf.exponents == (1.0, -1.0)
        assert isinstance(sf.exponents[0], float)


# =============================================================================
# Tier 2: Evaluation
# =============================================================================


class TestSingfunEval:
    """Tests for Singfun evaluation.

    JAX contract: __call__ jit=YES, vmap=YES, grad=YES.
    """

    def setup_method(self):
        self.sf_sqrt = Singfun.from_function(
            lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5)
        )
        self.sf_lsing = Singfun.from_function(
            lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0)
        )

    def test_scalar_eval(self):
        """Evaluate at scalar x=0."""
        val = float(self.sf_sqrt(jnp.float64(0.0)))
        npt.assert_allclose(val, 1.0, rtol=1e-14)

    def test_array_eval(self):
        """Evaluate at a batch of points."""
        x = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        vals = self.sf_sqrt(x)
        expected = jnp.sqrt(1.0 - x**2)
        npt.assert_allclose(np.array(vals), np.array(expected), rtol=1e-12)

    def test_left_sing_eval(self):
        """(1+x)^(-0.5) at interior point."""
        x = jnp.float64(0.5)
        val = float(self.sf_lsing(x))
        expected = float((1.0 + 0.5) ** (-0.5))
        npt.assert_allclose(val, expected, rtol=1e-13)

    def test_jit(self):
        """JIT-compiled evaluation via lambda wrapper."""
        sf = self.sf_sqrt
        jit_f = jax.jit(lambda x: sf(x))
        x = jnp.float64(0.3)
        npt.assert_allclose(
            float(jit_f(x)),
            float(sf(x)),
            rtol=1e-15,
        )

    def test_vmap(self):
        """vmap over evaluation points."""
        sf = self.sf_sqrt
        xs = jnp.linspace(-0.8, 0.8, 10, dtype=jnp.float64)
        vmap_vals = jax.vmap(sf)(xs)
        direct_vals = sf(xs)
        npt.assert_allclose(np.array(vmap_vals), np.array(direct_vals), rtol=1e-15)

    def test_grad(self):
        """Gradient of evaluation w.r.t. x."""
        sf = self.sf_sqrt
        # d/dx sqrt(1-x^2) = -x / sqrt(1-x^2)
        x0 = jnp.float64(0.3)
        g = float(jax.grad(lambda x: sf(x))(x0))
        expected = float(-0.3 / math.sqrt(1.0 - 0.3**2))
        npt.assert_allclose(g, expected, rtol=1e-10)


# =============================================================================
# Tier 3: Definite integral (sum)
# =============================================================================


class TestSingfunSum:
    """Tests for Singfun.sum (definite integral).

    JAX contract: sum jit=YES.

    Note: the Jacobi weight integral is computed correctly using
    Singfun.from_chebtech(Chebtech2([1.0]), (a, b)).
    Singfun.from_function(ones, (a, b)) represents the function 1.0 and
    its integral is always 2.
    """

    def test_sqrt_1mx2_integral(self):
        """integral of sqrt(1-x^2) from -1 to 1 = pi/2."""
        sf = Singfun.from_function(lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5))
        npt.assert_allclose(float(sf.sum()), math.pi / 2, rtol=1e-13)

    def test_1_over_sqrt_1mx2_integral(self):
        """integral of 1/sqrt(1-x^2) from -1 to 1 = pi.

        Constructed correctly: from_function(1/sqrt(1-x^2), (-0.5,-0.5))
        extracts smooth factor = sqrt(1-x^2) / [(1+x)^(-0.5)*(1-x)^(-0.5)]
        = (1-x^2) — wait that's wrong. Let's be careful:
        s(x) = f(x) / weight = [1/sqrt(1-x^2)] / [(1+x)^(-0.5)*(1-x)^(-0.5)]
             = [1/sqrt(1-x^2)] * sqrt(1-x^2) = 1.0
        So smooth factor is 1.0 and the integral = M_0 * 1 = pi.
        """
        sf = Singfun.from_function(
            lambda x: jnp.ones_like(x, dtype=jnp.float64) / jnp.sqrt(1.0 - x**2),
            (-0.5, -0.5),
        )
        npt.assert_allclose(float(sf.sum()), math.pi, rtol=1e-13)

    def test_left_sing_integral(self):
        """integral of (1+x)^(-0.5) from -1 to 1 = 2*sqrt(2)."""
        sf = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        npt.assert_allclose(
            float(sf.sum()), 2.0 * math.sqrt(2.0), rtol=1e-12
        )

    def test_right_sing_integral(self):
        """integral of (1-x)^(-0.5) from -1 to 1 = 2*sqrt(2)."""
        sf = Singfun.from_function(lambda x: (1.0 - x) ** (-0.5), (0.0, -0.5))
        npt.assert_allclose(
            float(sf.sum()), 2.0 * math.sqrt(2.0), rtol=1e-12
        )

    def test_trivial_exponents_delegates(self):
        """When exponents are (0,0), sum delegates to Chebtech2.sum."""
        sf = Singfun.from_function(lambda x: jnp.sin(x), (0.0, 0.0))
        npt.assert_allclose(float(sf.sum()), 0.0, atol=1e-13)

    def test_divergent_both_neg1(self):
        """Both exponents <= -1: integral diverges."""
        sf = Singfun.from_function(
            lambda x: jnp.ones_like(x, dtype=jnp.float64), (-1.0, -1.0)
        )
        result = float(sf.sum())
        assert jnp.isinf(result)

    def test_divergent_right_neg1(self):
        """Right exponent = -1: integral diverges (+inf)."""
        sf = Singfun.from_function(
            lambda x: jnp.ones_like(x, dtype=jnp.float64), (0.0, -1.0)
        )
        result = float(sf.sum())
        assert jnp.isinf(result) and result > 0

    def test_divergent_left_neg1(self):
        """Left exponent = -1: integral diverges (+inf)."""
        sf = Singfun.from_function(
            lambda x: jnp.ones_like(x, dtype=jnp.float64), (-1.0, 0.0)
        )
        result = float(sf.sum())
        assert jnp.isinf(result) and result > 0

    def test_jacobi_weight_from_chebtech(self):
        """Direct Jacobi weight integral via from_chebtech.

        integral of (1+x)^(1/3) (1-x)^(2/3) from -1 to 1
        = 2^(1/3+2/3+1) * B(4/3, 5/3) = 4 * B(4/3, 5/3).
        """
        a, b = 1.0 / 3.0, 2.0 / 3.0
        tech_one = Chebtech2.from_function(
            lambda x: jnp.ones_like(x, dtype=jnp.float64)
        )
        sf = Singfun.from_chebtech(tech_one, (a, b))
        expected = (2.0 ** (a + b + 1)) * (
            math.gamma(a + 1) * math.gamma(b + 1) / math.gamma(a + b + 2)
        )
        npt.assert_allclose(float(sf.sum()), expected, rtol=1e-11)

    def test_sum_jit(self):
        """sum is JIT-safe via lambda."""
        sf = Singfun.from_function(lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5))
        jit_sum = jax.jit(lambda: sf.sum())
        npt.assert_allclose(float(jit_sum()), math.pi / 2, rtol=1e-13)


# =============================================================================
# Tier 4: Differentiation
# =============================================================================


class TestSingfunDiff:
    """Tests for Singfun.diff.

    JAX contract: construction NOT jit-safe; result's __call__ IS jit-safe.
    """

    def test_diff_left_sing(self):
        """d/dx (1+x)^(-0.5) = -0.5 * (1+x)^(-1.5).

        At x=0: -0.5.
        """
        sf = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        dfdx = sf.diff()
        assert dfdx.exponents[0] == pytest.approx(-1.5, abs=1e-14)
        npt.assert_allclose(float(dfdx(jnp.float64(0.0))), -0.5, rtol=1e-12)

    def test_diff_smooth(self):
        """Derivative of sin(x) (smooth, exponents 0)."""
        sf = Singfun.from_function(jnp.sin, (0.0, 0.0))
        dfdx = sf.diff()
        x = jnp.linspace(-0.9, 0.9, 15, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(dfdx(x)), np.array(jnp.cos(x)), rtol=1e-12
        )

    def test_diff_k0(self):
        """diff(k=0) returns a copy."""
        sf = Singfun.from_function(lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5))
        sf0 = sf.diff(k=0)
        x = jnp.float64(0.3)
        npt.assert_allclose(float(sf0(x)), float(sf(x)), rtol=1e-15)

    def test_diff_k2(self):
        """Second derivative of (1+x)^(-0.5) = 0.75 * (1+x)^(-2.5).

        At x=0: 0.75.
        """
        sf = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        d2fdx2 = sf.diff(k=2)
        npt.assert_allclose(float(d2fdx2(jnp.float64(0.0))), 0.75, rtol=1e-10)

    def test_diff_result_jit(self):
        """Result of diff is JIT-safe for evaluation via lambda wrapper."""
        sf = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        dfdx = sf.diff()
        jit_eval = jax.jit(lambda x: dfdx(x))
        x = jnp.float64(0.5)
        npt.assert_allclose(float(jit_eval(x)), float(dfdx(x)), rtol=1e-15)


# =============================================================================
# Tier 5: Arithmetic
# =============================================================================


class TestSingfunArithmetic:
    """Tests for Singfun arithmetic.

    JAX contract: arithmetic NOT jit-safe at construction level.
    """

    def setup_method(self):
        self.sf1 = Singfun.from_function(
            lambda x: jnp.sqrt(1.0 - x**2), (0.5, 0.5)
        )
        self.sf2 = Singfun.from_function(
            lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0)
        )

    def test_scalar_mul(self):
        """f * c scales the smooth part."""
        sf3 = self.sf1 * 2.0
        x = jnp.array([0.0, 0.3, -0.5], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(sf3(x)),
            np.array(2.0 * self.sf1(x)),
            rtol=1e-14,
        )

    def test_scalar_rmul(self):
        """c * f scales the smooth part."""
        sf3 = 3.0 * self.sf1
        x = jnp.float64(0.4)
        npt.assert_allclose(float(sf3(x)), 3.0 * float(self.sf1(x)), rtol=1e-14)

    def test_neg(self):
        """Unary negation."""
        sf3 = -self.sf1
        x = jnp.float64(0.2)
        npt.assert_allclose(float(sf3(x)), -float(self.sf1(x)), rtol=1e-15)

    def test_singfun_mul(self):
        """f * g adds exponents and multiplies smooth parts."""
        sf3 = self.sf1 * self.sf2
        # (1+x)^0.5*(1-x)^0.5 * (1+x)^(-0.5) = (1+x)^0*(1-x)^0.5 = sqrt(1-x)
        # exponents: (0.5-0.5, 0.5+0.0) = (0.0, 0.5)
        assert sf3.exponents[0] == pytest.approx(0.0, abs=1e-15)
        assert sf3.exponents[1] == pytest.approx(0.5, abs=1e-15)
        x = jnp.array([0.0, 0.4], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(sf3(x)),
            np.array(jnp.sqrt(1.0 - x)),
            rtol=1e-12,
        )

    def test_pow(self):
        """f ** p raises exponents by p."""
        sf3 = self.sf2 ** 2  # (1+x)^(-0.5) squared -> (1+x)^(-1) * smooth^2
        assert sf3.exponents[0] == pytest.approx(-1.0, abs=1e-15)
        x = jnp.array([0.1, 0.3], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(sf3(x)),
            np.array((1.0 + x) ** (-1.0)),
            rtol=1e-12,
        )

    def test_add_same_exponents(self):
        """Addition with identical exponents just adds smooth parts."""
        sf3 = self.sf1 + self.sf1
        x = jnp.float64(0.3)
        npt.assert_allclose(float(sf3(x)), 2.0 * float(self.sf1(x)), rtol=1e-13)
        assert sf3.exponents == self.sf1.exponents

    def test_add_integer_diff_exponents(self):
        """Addition with integer-differing exponents is handled exactly."""
        # (1+x)^(-0.5) + (1+x)^(0.5)
        sf_a = Singfun.from_function(lambda x: (1.0 + x) ** (-0.5), (-0.5, 0.0))
        sf_b = Singfun.from_function(lambda x: (1.0 + x) ** (0.5), (0.5, 0.0))
        sf_sum = sf_a + sf_b
        x = jnp.array([0.1, 0.3, 0.5], dtype=jnp.float64)
        expected = (1.0 + x) ** (-0.5) + (1.0 + x) ** (0.5)
        npt.assert_allclose(
            np.array(sf_sum(x)), np.array(expected), rtol=1e-10
        )

    def test_sub(self):
        """Subtraction."""
        sf3 = self.sf1 - self.sf1
        x = jnp.float64(0.3)
        npt.assert_allclose(abs(float(sf3(x))), 0.0, atol=1e-13)

    def test_truediv_scalar(self):
        """f / scalar."""
        sf3 = self.sf1 / 2.0
        x = jnp.float64(0.3)
        npt.assert_allclose(float(sf3(x)), float(self.sf1(x)) / 2.0, rtol=1e-14)
