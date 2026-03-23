"""Tests for chebfunjax.utils.polynomials — classical orthogonal polynomials.

JAX contract: jit=yes (n and parameters must be static), vmap=no, grad=yes
"""

import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.polynomials import (
    chebeval,
    chebpoly,
    hermeval,
    jaceval,
    jacpoly,
    lageval,
    legeval,
    legpoly,
    ultraeval,
    ultrapoly,
)

# ===========================================================================
# Tier 1: Pure mathematical tests (no MATLAB)
# ===========================================================================


class TestChebpoly:
    """Tests for chebpoly — Chebyshev polynomial coefficient vectors."""

    def test_T0(self):
        """T_0 = 1: coefficients [1]."""
        c = chebpoly(0)
        npt.assert_allclose(np.array(c), [1.0], atol=1e-15)

    def test_T1(self):
        """T_1 = x: coefficients [0, 1]."""
        c = chebpoly(1)
        npt.assert_allclose(np.array(c), [0.0, 1.0], atol=1e-15)

    def test_T2(self):
        """T_2 = 2x^2 - 1: coefficients [0, 0, 1]."""
        c = chebpoly(2)
        npt.assert_allclose(np.array(c), [0.0, 0.0, 1.0], atol=1e-15)

    def test_T5(self):
        """T_5: coefficients are zero except at position 5."""
        c = chebpoly(5)
        expected = np.zeros(6)
        expected[5] = 1.0
        npt.assert_allclose(np.array(c), expected, atol=1e-15)

    def test_shape(self):
        """Output shape is (n+1,)."""
        for n in [0, 1, 5, 10, 50]:
            c = chebpoly(n)
            assert c.shape == (n + 1,), f"Expected shape ({n + 1},), got {c.shape}"

    def test_U0(self):
        """U_0 = 1 = T_0: coefficients [1]."""
        c = chebpoly(0, kind=2)
        npt.assert_allclose(np.array(c), [1.0], atol=1e-15)

    def test_U1(self):
        """U_1 = 2x = 2*T_1: coefficients [0, 2]."""
        c = chebpoly(1, kind=2)
        npt.assert_allclose(np.array(c), [0.0, 2.0], atol=1e-15)

    def test_U2(self):
        """U_2 = 4x^2 - 1 = T_0 + 2*T_2: coefficients [1, 0, 2]."""
        c = chebpoly(2, kind=2)
        npt.assert_allclose(np.array(c), [1.0, 0.0, 2.0], atol=1e-15)

    def test_U3(self):
        """U_3 = 8x^3 - 4x = 2*T_1 + 2*T_3: coefficients [0, 2, 0, 2]."""
        c = chebpoly(3, kind=2)
        npt.assert_allclose(np.array(c), [0.0, 2.0, 0.0, 2.0], atol=1e-15)

    def test_U4(self):
        """U_4: coefficients [1, 0, 2, 0, 2]."""
        c = chebpoly(4, kind=2)
        npt.assert_allclose(np.array(c), [1.0, 0.0, 2.0, 0.0, 2.0], atol=1e-15)

    def test_U_evaluation(self):
        """Verify U_n coefficients evaluate correctly at sample points."""
        x = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        for n in [0, 1, 2, 3, 5, 8]:
            c = chebpoly(n, kind=2)
            # Evaluate via Clenshaw (using coeffs2vals would need grid points)
            # Instead use chebeval for U_n
            vals_direct = chebeval(x, n, kind=2)
            # Evaluate from coefficients: sum c_k * T_k(x)
            vals_from_coeffs = jnp.zeros_like(x)
            for k in range(len(c)):
                vals_from_coeffs = vals_from_coeffs + c[k] * chebeval(x, k, kind=1)
            npt.assert_allclose(
                np.array(vals_from_coeffs), np.array(vals_direct),
                rtol=1e-13, atol=1e-14,
                err_msg=f"U_{n} coefficient evaluation mismatch"
            )

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="non-negative"):
            chebpoly(-1)

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="kind must be"):
            chebpoly(3, kind=3)


class TestChebeval:
    """Tests for chebeval — Chebyshev polynomial evaluation."""

    def test_T0(self):
        """T_0(x) = 1 for all x."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(chebeval(x, 0)), 1.0, atol=1e-15)

    def test_T1(self):
        """T_1(x) = x."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(chebeval(x, 1)), np.array(x), atol=1e-15)

    def test_T2(self):
        """T_2(x) = 2x^2 - 1."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = 2 * x**2 - 1
        npt.assert_allclose(np.array(chebeval(x, 2)), np.array(expected), atol=1e-14)

    def test_T_at_1(self):
        """T_n(1) = 1 for all n."""
        x = jnp.array(1.0, dtype=jnp.float64)
        for n in range(10):
            npt.assert_allclose(float(chebeval(x, n)), 1.0, atol=1e-14)

    def test_T_at_neg1(self):
        """T_n(-1) = (-1)^n for all n."""
        x = jnp.array(-1.0, dtype=jnp.float64)
        for n in range(10):
            npt.assert_allclose(float(chebeval(x, n)), (-1.0)**n, atol=1e-14)

    def test_U0(self):
        """U_0(x) = 1."""
        x = jnp.linspace(-0.99, 0.99, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(chebeval(x, 0, kind=2)), 1.0, atol=1e-14)

    def test_U1(self):
        """U_1(x) = 2x."""
        x = jnp.linspace(-0.99, 0.99, 50, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(chebeval(x, 1, kind=2)),
            np.array(2 * x),
            atol=1e-14
        )

    def test_U2(self):
        """U_2(x) = 4x^2 - 1."""
        x = jnp.linspace(-0.99, 0.99, 50, dtype=jnp.float64)
        expected = 4 * x**2 - 1
        npt.assert_allclose(
            np.array(chebeval(x, 2, kind=2)),
            np.array(expected),
            atol=1e-13
        )

    def test_U_at_1(self):
        """U_n(1) = n+1."""
        x = jnp.array(1.0, dtype=jnp.float64)
        for n in range(10):
            npt.assert_allclose(
                float(chebeval(x, n, kind=2)),
                float(n + 1),
                atol=1e-12
            )


class TestLegpoly:
    """Tests for legpoly — Legendre polynomial Chebyshev coefficients."""

    def test_P0(self):
        """P_0 = 1 = T_0."""
        c = legpoly(0)
        npt.assert_allclose(np.array(c), [1.0], atol=1e-15)

    def test_P1(self):
        """P_1 = x = T_1."""
        c = legpoly(1)
        npt.assert_allclose(np.array(c), [0.0, 1.0], atol=1e-15)

    def test_P2(self):
        """P_2 = (3x^2-1)/2 = 1/4*T_0 + 3/4*T_2."""
        c = legpoly(2)
        npt.assert_allclose(np.array(c), [0.25, 0.0, 0.75], atol=1e-14)

    def test_P3(self):
        """P_3 = (5x^3-3x)/2 = 3/8*T_1 + 5/8*T_3."""
        c = legpoly(3)
        npt.assert_allclose(
            np.array(c), [0.0, 3.0 / 8, 0.0, 5.0 / 8], atol=1e-14
        )

    def test_normalized_integral_1(self):
        """Normalized P_0 should integrate to 1 (i.e., int P_0^2 dx = 1)."""
        c = legpoly(0, normalize=True)
        # P_0 normalized: 1 / sqrt(2), so Cheb coeff is 1/sqrt(2)
        npt.assert_allclose(np.array(c), [1.0 / jnp.sqrt(2.0)], rtol=1e-14)

    def test_evaluation_matches_legeval(self):
        """Chebyshev coefficients from legpoly evaluate to same as legeval."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        for n in [0, 1, 2, 3, 5, 8, 15]:
            c = legpoly(n)
            # Evaluate Chebyshev expansion at x
            vals_cheb = jnp.zeros_like(x)
            for k in range(len(c)):
                vals_cheb = vals_cheb + c[k] * chebeval(x, k)
            vals_direct = legeval(x, n)
            npt.assert_allclose(
                np.array(vals_cheb), np.array(vals_direct),
                rtol=1e-12, atol=1e-13,
                err_msg=f"P_{n} mismatch"
            )

    def test_invalid_n(self):
        with pytest.raises(ValueError, match="non-negative"):
            legpoly(-1)


class TestLegeval:
    """Tests for legeval — Legendre polynomial evaluation."""

    def test_P0(self):
        """P_0(x) = 1."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(legeval(x, 0)), 1.0, atol=1e-15)

    def test_P1(self):
        """P_1(x) = x."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(legeval(x, 1)), np.array(x), atol=1e-15)

    def test_P2(self):
        """P_2(x) = (3x^2-1)/2."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = (3 * x**2 - 1) / 2
        npt.assert_allclose(np.array(legeval(x, 2)), np.array(expected), atol=1e-14)

    def test_P_at_1(self):
        """P_n(1) = 1 for all n."""
        x = jnp.array(1.0, dtype=jnp.float64)
        for n in range(15):
            npt.assert_allclose(
                float(legeval(x, n)), 1.0, atol=1e-13,
                err_msg=f"P_{n}(1) != 1"
            )

    def test_P_at_neg1(self):
        """P_n(-1) = (-1)^n."""
        x = jnp.array(-1.0, dtype=jnp.float64)
        for n in range(15):
            npt.assert_allclose(
                float(legeval(x, n)), (-1.0)**n, atol=1e-13,
                err_msg=f"P_{n}(-1) != (-1)^{n}"
            )

    def test_orthogonality(self):
        """Integral of P_m * P_n over [-1,1] = 2/(2n+1) delta_{mn}."""
        from chebfunjax.utils.quadrature import chebpts, chebweights

        N = 100  # quadrature order
        x = chebpts(N, kind=2)
        w = chebweights(N, kind=2)
        for m in range(6):
            Pm = legeval(x, m)
            for n in range(6):
                Pn = legeval(x, n)
                integral = float(jnp.dot(w, Pm * Pn))
                if m == n:
                    expected = 2.0 / (2 * n + 1)
                else:
                    expected = 0.0
                npt.assert_allclose(
                    integral, expected, atol=1e-12,
                    err_msg=f"Orthogonality failed for (m={m}, n={n})"
                )


class TestJacpoly:
    """Tests for jacpoly — Jacobi polynomial Chebyshev coefficients."""

    def test_P0(self):
        """P_0^{(a,b)} = 1 for any (a,b)."""
        for a, b in [(0.5, 0.5), (1.0, 2.0), (0.1, 0.3)]:
            c = jacpoly(0, a, b)
            npt.assert_allclose(np.array(c), [1.0], atol=1e-14)

    def test_legendre_case(self):
        """jacpoly(n, 0, 0) should match legpoly(n)."""
        for n in [0, 1, 2, 3, 5, 10]:
            c_jac = jacpoly(n, 0.0, 0.0)
            c_leg = legpoly(n)
            npt.assert_allclose(
                np.array(c_jac), np.array(c_leg),
                rtol=1e-12, atol=1e-14,
                err_msg=f"jacpoly({n}, 0, 0) != legpoly({n})"
            )

    def test_evaluation_matches_jaceval(self):
        """Chebyshev coefficients from jacpoly evaluate correctly."""
        x = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        for n in [0, 1, 2, 3, 5]:
            for a, b in [(0.5, 0.5), (1.0, 0.5)]:
                c = jacpoly(n, a, b)
                vals_cheb = jnp.zeros_like(x)
                for k in range(len(c)):
                    vals_cheb = vals_cheb + c[k] * chebeval(x, k)
                vals_direct = jaceval(x, n, a, b)
                npt.assert_allclose(
                    np.array(vals_cheb), np.array(vals_direct),
                    rtol=1e-11, atol=1e-12,
                    err_msg=f"jacpoly({n}, {a}, {b}) eval mismatch"
                )


class TestJaceval:
    """Tests for jaceval — Jacobi polynomial evaluation."""

    def test_P0(self):
        """P_0^{(a,b)}(x) = 1."""
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(jaceval(x, 0, 0.5, 0.5)), 1.0, atol=1e-15)

    def test_P1(self):
        """P_1^{(a,b)}(x) = (a+1) + (a+b+2)*(x-1)/2."""
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        a, b = 1.5, 0.7
        expected = (a + 1.0) + (a + b + 2.0) * (x - 1.0) / 2.0
        npt.assert_allclose(
            np.array(jaceval(x, 1, a, b)),
            np.array(expected),
            rtol=1e-14
        )

    def test_P_at_1(self):
        """P_n^{(a,b)}(1) = C(n+a, n) = rising_factorial(a+1, n) / n!."""
        from scipy.special import poch

        x = jnp.array(1.0, dtype=jnp.float64)
        for n in range(8):
            for a, b in [(0.5, 0.5), (1.0, 2.0), (0.3, 0.7)]:
                val = float(jaceval(x, n, a, b))
                expected = poch(a + 1, n) / math.factorial(n) if n > 0 else 1.0
                npt.assert_allclose(
                    val, expected, rtol=1e-12,
                    err_msg=f"P_{n}^({a},{b})(1) wrong"
                )

    def test_legendre_case(self):
        """P_n^{(0,0)} = P_n (Legendre)."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        for n in [0, 1, 2, 5, 10]:
            npt.assert_allclose(
                np.array(jaceval(x, n, 0.0, 0.0)),
                np.array(legeval(x, n)),
                rtol=1e-13, atol=1e-14,
                err_msg=f"P_{n}^(0,0) != P_{n}"
            )


class TestUltrapoly:
    """Tests for ultrapoly — ultraspherical polynomial Chebyshev coefficients."""

    def test_C0(self):
        """C_0^{(lam)} = 1 for any lam > 0."""
        for lam in [0.5, 1.0, 2.0]:
            c = ultrapoly(0, lam)
            npt.assert_allclose(np.array(c), [1.0], atol=1e-14)

    def test_legendre_special_case(self):
        """ultrapoly(n, 0.5) should match legpoly(n)."""
        for n in [0, 1, 2, 3, 5]:
            c_ultra = ultrapoly(n, 0.5)
            c_leg = legpoly(n)
            npt.assert_allclose(
                np.array(c_ultra), np.array(c_leg),
                rtol=1e-12, atol=1e-14,
                err_msg=f"ultrapoly({n}, 0.5) != legpoly({n})"
            )

    def test_chebyshev_U_special_case(self):
        """ultrapoly(n, 1.0) should match chebpoly(n, kind=2)."""
        for n in [0, 1, 2, 3, 5]:
            c_ultra = ultrapoly(n, 1.0)
            c_cheb2 = chebpoly(n, kind=2)
            npt.assert_allclose(
                np.array(c_ultra), np.array(c_cheb2),
                rtol=1e-12, atol=1e-14,
                err_msg=f"ultrapoly({n}, 1) != chebpoly({n}, kind=2)"
            )

    def test_evaluation_matches_ultraeval(self):
        """Chebyshev coefficients from ultrapoly evaluate correctly."""
        x = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        for n in [0, 1, 2, 3, 5]:
            for lam in [0.5, 1.0, 1.5, 2.0]:
                c = ultrapoly(n, lam)
                vals_cheb = jnp.zeros_like(x)
                for k in range(len(c)):
                    vals_cheb = vals_cheb + c[k] * chebeval(x, k)
                vals_direct = ultraeval(x, n, lam)
                npt.assert_allclose(
                    np.array(vals_cheb), np.array(vals_direct),
                    rtol=1e-11, atol=1e-12,
                    err_msg=f"ultrapoly({n}, {lam}) eval mismatch"
                )

    def test_invalid_lam(self):
        with pytest.raises(ValueError, match="positive"):
            ultrapoly(3, -0.5)


class TestUltraeval:
    """Tests for ultraeval — ultraspherical polynomial evaluation."""

    def test_C0(self):
        """C_0^{(lam)}(x) = 1."""
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(ultraeval(x, 0, 1.5)), 1.0, atol=1e-15)

    def test_C1(self):
        """C_1^{(lam)}(x) = 2*lam*x."""
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        lam = 2.5
        npt.assert_allclose(
            np.array(ultraeval(x, 1, lam)),
            np.array(2 * lam * x),
            rtol=1e-14
        )

    def test_C_at_1(self):
        """C_n^{(lam)}(1) = (2*lam)_n / n!."""
        from scipy.special import poch

        x = jnp.array(1.0, dtype=jnp.float64)
        for n in range(8):
            for lam in [0.5, 1.0, 1.5, 2.0]:
                val = float(ultraeval(x, n, lam))
                expected = poch(2 * lam, n) / math.factorial(n) if n > 0 else 1.0
                npt.assert_allclose(
                    val, expected, rtol=1e-12,
                    err_msg=f"C_{n}^({lam})(1) wrong"
                )

    def test_legendre_case(self):
        """C_n^{(0.5)} = P_n (Legendre)."""
        x = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        for n in [0, 1, 2, 5, 10]:
            npt.assert_allclose(
                np.array(ultraeval(x, n, 0.5)),
                np.array(legeval(x, n)),
                rtol=1e-13, atol=1e-14,
                err_msg=f"C_{n}^(0.5) != P_{n}"
            )


class TestHermeval:
    """Tests for hermeval — Hermite polynomial evaluation."""

    def test_H0_phys(self):
        """H_0(x) = 1 (physicist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(hermeval(x, 0)), 1.0, atol=1e-15)

    def test_H1_phys(self):
        """H_1(x) = 2x (physicist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(hermeval(x, 1)),
            np.array(2 * x),
            atol=1e-14
        )

    def test_H2_phys(self):
        """H_2(x) = 4x^2 - 2 (physicist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        expected = 4 * x**2 - 2
        npt.assert_allclose(
            np.array(hermeval(x, 2)),
            np.array(expected),
            rtol=1e-14
        )

    def test_H3_phys(self):
        """H_3(x) = 8x^3 - 12x (physicist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        expected = 8 * x**3 - 12 * x
        npt.assert_allclose(
            np.array(hermeval(x, 3)),
            np.array(expected),
            rtol=1e-13
        )

    def test_He0_prob(self):
        """He_0(x) = 1 (probabilist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(hermeval(x, 0, kind="prob")),
            1.0,
            atol=1e-15
        )

    def test_He1_prob(self):
        """He_1(x) = x (probabilist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        npt.assert_allclose(
            np.array(hermeval(x, 1, kind="prob")),
            np.array(x),
            atol=1e-14
        )

    def test_He2_prob(self):
        """He_2(x) = x^2 - 1 (probabilist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        expected = x**2 - 1
        npt.assert_allclose(
            np.array(hermeval(x, 2, kind="prob")),
            np.array(expected),
            rtol=1e-14
        )

    def test_He3_prob(self):
        """He_3(x) = x^3 - 3x (probabilist)."""
        x = jnp.linspace(-3, 3, 50, dtype=jnp.float64)
        expected = x**3 - 3 * x
        npt.assert_allclose(
            np.array(hermeval(x, 3, kind="prob")),
            np.array(expected),
            rtol=1e-13
        )

    def test_orthogonality_phys(self):
        """Integral of H_m * H_n * exp(-x^2) ~ 2^n * n! * sqrt(pi) * delta_{mn}."""
        # Use Gauss-Hermite quadrature from numpy
        from numpy.polynomial.hermite import hermgauss

        pts, wts = hermgauss(40)
        x = jnp.array(pts, dtype=jnp.float64)
        # hermgauss weights already include exp(-x^2)
        for m in range(5):
            Hm = hermeval(x, m)
            for n in range(5):
                Hn = hermeval(x, n)
                integral = float(jnp.dot(jnp.array(wts, dtype=jnp.float64), Hm * Hn))
                if m == n:
                    expected = 2**n * math.factorial(n) * np.sqrt(np.pi)
                else:
                    expected = 0.0
                npt.assert_allclose(
                    integral, expected, atol=1e-10,
                    err_msg=f"Hermite orthogonality failed for (m={m}, n={n})"
                )

    def test_invalid_kind(self):
        x = jnp.array([0.0])
        with pytest.raises(ValueError, match="kind must be"):
            hermeval(x, 3, kind="bogus")

    def test_scipy_consistency(self):
        """Compare against scipy.special.hermite (physicist's)."""
        from scipy.special import hermite

        x = np.linspace(-3, 3, 50)
        for n in range(8):
            scipy_vals = hermite(n)(x)
            jax_vals = np.array(hermeval(jnp.array(x, dtype=jnp.float64), n))
            npt.assert_allclose(
                jax_vals, scipy_vals, rtol=1e-12, atol=1e-12,
                err_msg=f"H_{n} scipy mismatch"
            )


class TestLageval:
    """Tests for lageval — Laguerre polynomial evaluation."""

    def test_L0(self):
        """L_0(x) = 1."""
        x = jnp.linspace(0, 10, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(lageval(x, 0)), 1.0, atol=1e-15)

    def test_L1(self):
        """L_1(x) = 1 - x."""
        x = jnp.linspace(0, 10, 50, dtype=jnp.float64)
        expected = 1.0 - x
        npt.assert_allclose(
            np.array(lageval(x, 1)),
            np.array(expected),
            atol=1e-14
        )

    def test_L2(self):
        """L_2(x) = (x^2 - 4x + 2) / 2."""
        x = jnp.linspace(0, 10, 50, dtype=jnp.float64)
        expected = (x**2 - 4 * x + 2) / 2
        npt.assert_allclose(
            np.array(lageval(x, 2)),
            np.array(expected),
            rtol=1e-14
        )

    def test_L3(self):
        """L_3(x) = (-x^3 + 9x^2 - 18x + 6) / 6."""
        x = jnp.linspace(0, 10, 50, dtype=jnp.float64)
        expected = (-x**3 + 9 * x**2 - 18 * x + 6) / 6
        npt.assert_allclose(
            np.array(lageval(x, 3)),
            np.array(expected),
            rtol=1e-12
        )

    def test_generalized_L0(self):
        """L_0^{(alpha)}(x) = 1 for any alpha."""
        x = jnp.linspace(0, 10, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(lageval(x, 0, alpha=2.5)), 1.0, atol=1e-15)

    def test_generalized_L1(self):
        """L_1^{(alpha)}(x) = 1 + alpha - x."""
        x = jnp.linspace(0, 10, 20, dtype=jnp.float64)
        alpha = 2.5
        expected = 1 + alpha - x
        npt.assert_allclose(
            np.array(lageval(x, 1, alpha=alpha)),
            np.array(expected),
            atol=1e-14
        )

    def test_orthogonality(self):
        """Integral of L_m * L_n * exp(-x) over [0,inf] = delta_{mn}."""
        # Use Gauss-Laguerre quadrature
        from numpy.polynomial.laguerre import laggauss

        pts, wts = laggauss(40)
        x = jnp.array(pts, dtype=jnp.float64)
        # laggauss weights include exp(-x)
        for m in range(5):
            Lm = lageval(x, m)
            for n in range(5):
                Ln = lageval(x, n)
                integral = float(jnp.dot(jnp.array(wts, dtype=jnp.float64), Lm * Ln))
                if m == n:
                    expected = 1.0
                else:
                    expected = 0.0
                npt.assert_allclose(
                    integral, expected, atol=1e-12,
                    err_msg=f"Laguerre orthogonality failed for (m={m}, n={n})"
                )

    def test_scipy_consistency(self):
        """Compare against scipy.special.eval_laguerre."""
        from scipy.special import eval_laguerre

        x_np = np.linspace(0, 10, 50)
        x = jnp.array(x_np, dtype=jnp.float64)
        for n in range(8):
            scipy_vals = eval_laguerre(n, x_np)
            jax_vals = np.array(lageval(x, n))
            npt.assert_allclose(
                jax_vals, scipy_vals, rtol=1e-12,
                err_msg=f"L_{n} scipy mismatch"
            )

    def test_generalized_scipy_consistency(self):
        """Compare generalized Laguerre against scipy.special.eval_genlaguerre."""
        from scipy.special import eval_genlaguerre

        x_np = np.linspace(0, 10, 50)
        x = jnp.array(x_np, dtype=jnp.float64)
        for n in range(6):
            for alpha in [0.5, 1.0, 2.5]:
                scipy_vals = eval_genlaguerre(n, alpha, x_np)
                jax_vals = np.array(lageval(x, n, alpha=alpha))
                npt.assert_allclose(
                    jax_vals, scipy_vals, rtol=1e-12,
                    err_msg=f"L_{n}^({alpha}) scipy mismatch"
                )


class TestChebyshevOrthogonality:
    """Test orthogonality of Chebyshev polynomials."""

    def test_T_orthogonality(self):
        """Integral of T_m * T_n / sqrt(1-x^2) dx = {0, pi/2, pi} for m!=n, m=n>0, m=n=0."""
        from chebfunjax.utils.quadrature import chebpts, chebweights

        # Gauss-Chebyshev quadrature (1st kind): weights = pi/n
        N = 100
        x = chebpts(N, kind=1)
        w = chebweights(N, kind=1)
        for m in range(6):
            Tm = chebeval(x, m)
            for n in range(6):
                Tn = chebeval(x, n)
                # Gauss-Chebyshev: sum w_k * f(x_k) approximates int f(x)/sqrt(1-x^2) dx
                integral = float(jnp.dot(w, Tm * Tn))
                if m != n:
                    npt.assert_allclose(
                        integral, 0.0, atol=1e-12,
                        err_msg=f"T_{m}*T_{n} not orthogonal"
                    )
                elif m == 0:
                    npt.assert_allclose(
                        integral, jnp.pi, atol=1e-12,
                        err_msg="T_0*T_0 integral wrong"
                    )
                else:
                    npt.assert_allclose(
                        integral, jnp.pi / 2, atol=1e-12,
                        err_msg=f"T_{m}*T_{m} integral wrong"
                    )


# ===========================================================================
# Tier 2: MATLAB cross-validation
# ===========================================================================


class TestPolynomialsMATLAB:
    """Compare against MATLAB Chebfun reference data."""

    @pytest.mark.matlab
    def test_chebpoly_vs_matlab(self, matlab_polynomials):
        """Chebyshev polynomial coefficients match MATLAB."""
        for n in [0, 1, 2, 5, 10, 20]:
            ref = matlab_polynomials[f"chebpoly_T{n}"]
            c = np.array(chebpoly(n))
            npt.assert_allclose(
                c, ref, atol=1e-14,
                err_msg=f"chebpoly(T_{n}) MATLAB mismatch"
            )

    @pytest.mark.matlab
    def test_chebpoly_U_vs_matlab(self, matlab_polynomials):
        """Second-kind Chebyshev polynomial coefficients match MATLAB."""
        for n in [0, 1, 2, 5, 10]:
            ref = matlab_polynomials[f"chebpoly_U{n}"]
            c = np.array(chebpoly(n, kind=2))
            npt.assert_allclose(
                c, ref, atol=1e-14,
                err_msg=f"chebpoly(U_{n}) MATLAB mismatch"
            )

    @pytest.mark.matlab
    def test_legpoly_vs_matlab(self, matlab_polynomials):
        """Legendre polynomial Chebyshev coefficients match MATLAB."""
        for n in [0, 1, 2, 5, 10, 20]:
            ref = matlab_polynomials[f"legpoly_P{n}"]
            c = np.array(legpoly(n))
            npt.assert_allclose(
                c, ref, rtol=1e-12, atol=1e-14,
                err_msg=f"legpoly(P_{n}) MATLAB mismatch"
            )

    @pytest.mark.matlab
    def test_jacpoly_vs_matlab(self, matlab_polynomials):
        """Jacobi polynomial Chebyshev coefficients match MATLAB."""
        for n in [0, 1, 2, 5, 10]:
            for a_str, a_val, b_str, b_val in [
                ("0p5", 0.5, "0p5", 0.5),
                ("1p0", 1.0, "0p5", 0.5),
                ("2p0", 2.0, "1p5", 1.5),
            ]:
                key = f"jacpoly_n{n}_a{a_str}_b{b_str}"
                if key in matlab_polynomials:
                    ref = matlab_polynomials[key]
                    c = np.array(jacpoly(n, a_val, b_val))
                    npt.assert_allclose(
                        c, ref, rtol=1e-11, atol=1e-13,
                        err_msg=f"jacpoly({n}, {a_val}, {b_val}) MATLAB mismatch"
                    )

    @pytest.mark.matlab
    def test_ultrapoly_vs_matlab(self, matlab_polynomials):
        """Ultraspherical polynomial Chebyshev coefficients match MATLAB."""
        for n in [0, 1, 2, 5, 10]:
            for lam_str, lam_val in [("1p5", 1.5), ("2p0", 2.0)]:
                key = f"ultrapoly_n{n}_lam{lam_str}"
                if key in matlab_polynomials:
                    ref = matlab_polynomials[key]
                    c = np.array(ultrapoly(n, lam_val))
                    npt.assert_allclose(
                        c, ref, rtol=1e-11, atol=1e-13,
                        err_msg=f"ultrapoly({n}, {lam_val}) MATLAB mismatch"
                    )

    @pytest.mark.matlab
    def test_hermeval_vs_matlab(self, matlab_polynomials):
        """Hermite polynomial values match MATLAB.

        Note: MATLAB hermpoly uses Chebfun adaptive construction on [-inf, inf]
        with barycentric interpolation and mapping, which the MATLAB source itself
        calls a "toy". Our direct recurrence matches scipy exactly and is more
        accurate than MATLAB's representation for near-zero values. We use a
        looser tolerance accordingly (rtol=1e-10 vs the default 1e-12).
        """
        ref_x = matlab_polynomials.get("hermeval_x")
        if ref_x is not None:
            x = jnp.array(ref_x, dtype=jnp.float64)
            for n in [0, 1, 2, 3, 5]:
                key = f"hermeval_H{n}"
                if key in matlab_polynomials:
                    ref = matlab_polynomials[key]
                    vals = np.array(hermeval(x, n))
                    npt.assert_allclose(
                        vals, ref, rtol=1e-10, atol=1e-10,
                        err_msg=f"hermeval(H_{n}) MATLAB mismatch"
                    )

    @pytest.mark.matlab
    def test_lageval_vs_matlab(self, matlab_polynomials):
        """Laguerre polynomial values match MATLAB.

        Note: MATLAB lagpoly uses Chebfun adaptive construction on [0, inf]
        with barycentric interpolation and mapping. Like hermpoly, the MATLAB
        source calls it a "toy". Our direct recurrence matches scipy exactly.
        We use rtol=1e-10 to account for MATLAB interpolation errors on
        unbounded domains (worst case: ~2.4e-11 for L_5 near zero crossings).
        """
        ref_x = matlab_polynomials.get("lageval_x")
        if ref_x is not None:
            x = jnp.array(ref_x, dtype=jnp.float64)
            for n in [0, 1, 2, 3, 5]:
                key = f"lageval_L{n}"
                if key in matlab_polynomials:
                    ref = matlab_polynomials[key]
                    vals = np.array(lageval(x, n))
                    npt.assert_allclose(
                        vals, ref, rtol=1e-10, atol=1e-10,
                        err_msg=f"lageval(L_{n}) MATLAB mismatch"
                    )


# ===========================================================================
# JIT compatibility tests
# ===========================================================================


class TestJITCompatibility:
    """Verify polynomial functions work under jax.jit.

    JAX contract: jit=yes (n must be static), vmap=no, grad=yes
    """

    def test_chebpoly_jit(self):
        jitted = jax.jit(functools.partial(chebpoly, 5, kind=1))
        npt.assert_allclose(
            np.array(jitted()), np.array(chebpoly(5)), rtol=1e-15
        )

    def test_chebeval_jit(self):
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(chebeval, n=5, kind=1))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(chebeval(x, 5)), rtol=1e-15
        )

    def test_legpoly_jit(self):
        jitted = jax.jit(functools.partial(legpoly, 5))
        npt.assert_allclose(
            np.array(jitted()), np.array(legpoly(5)), rtol=1e-14, atol=1e-15
        )

    def test_legeval_jit(self):
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(legeval, n=5))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(legeval(x, 5)), rtol=1e-14
        )

    def test_jacpoly_jit(self):
        jitted = jax.jit(functools.partial(jacpoly, 5, alpha=0.5, beta=0.5))
        npt.assert_allclose(
            np.array(jitted()), np.array(jacpoly(5, 0.5, 0.5)), rtol=1e-13, atol=1e-14
        )

    def test_jaceval_jit(self):
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(jaceval, n=5, alpha=0.5, beta=0.5))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(jaceval(x, 5, 0.5, 0.5)), rtol=1e-13
        )

    def test_hermeval_jit(self):
        x = jnp.linspace(-3, 3, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(hermeval, n=5, kind="phys"))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(hermeval(x, 5)), rtol=1e-14
        )

    def test_lageval_jit(self):
        x = jnp.linspace(0, 10, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(lageval, n=5, alpha=0.0))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(lageval(x, 5)), rtol=1e-14
        )

    def test_ultrapoly_jit(self):
        jitted = jax.jit(functools.partial(ultrapoly, 5, lam=1.5))
        npt.assert_allclose(
            np.array(jitted()), np.array(ultrapoly(5, 1.5)), rtol=1e-13, atol=1e-13
        )

    def test_ultraeval_jit(self):
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        jitted = jax.jit(functools.partial(ultraeval, n=5, lam=1.5))
        npt.assert_allclose(
            np.array(jitted(x)), np.array(ultraeval(x, 5, 1.5)), rtol=1e-13
        )


# ===========================================================================
# Grad compatibility tests
# ===========================================================================


class TestGradCompatibility:
    """Verify evaluation functions are differentiable."""

    def test_chebeval_grad(self):
        """Gradient of T_n w.r.t. x should be n*U_{n-1}."""
        x0 = jnp.array(0.5, dtype=jnp.float64)
        for n in [1, 2, 3, 5]:
            g = jax.grad(lambda x: chebeval(x, n, kind=1))(x0)
            # T_n'(x) = n * U_{n-1}(x)
            expected = n * chebeval(x0, n - 1, kind=2)
            npt.assert_allclose(
                float(g), float(expected), rtol=1e-11,
                err_msg=f"T_{n}'(0.5) grad wrong"
            )

    def test_legeval_grad(self):
        """Gradient of P_n evaluation should be finite and correct shape."""
        x0 = jnp.array(0.5, dtype=jnp.float64)
        g = jax.grad(lambda x: legeval(x, 3))(x0)
        # P_3'(x) = (5x^2 - 1) * 3/2
        # Actually P_3 = (5x^3 - 3x)/2, so P_3' = (15x^2 - 3)/2
        expected = (15 * 0.5**2 - 3) / 2
        npt.assert_allclose(float(g), expected, rtol=1e-12)

    def test_hermeval_grad(self):
        """H_n'(x) = 2n * H_{n-1}(x) (physicist's)."""
        x0 = jnp.array(0.7, dtype=jnp.float64)
        for n in [1, 2, 3, 5]:
            g = jax.grad(lambda x: hermeval(x, n))(x0)
            expected = 2.0 * n * hermeval(x0, n - 1)
            npt.assert_allclose(
                float(g), float(expected), rtol=1e-11,
                err_msg=f"H_{n}'(0.7) grad wrong"
            )

    def test_lageval_grad_numerical(self):
        """Numerical gradient check for lageval."""
        x0 = jnp.array(1.5, dtype=jnp.float64)
        eps = 1e-7
        for n in [1, 2, 3, 5]:
            g = jax.grad(lambda x: lageval(x, n))(x0)
            g_num = (float(lageval(x0 + eps, n)) - float(lageval(x0 - eps, n))) / (2 * eps)
            npt.assert_allclose(
                float(g), g_num, rtol=1e-5, atol=1e-10,
                err_msg=f"L_{n}'(1.5) numerical grad mismatch"
            )


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def matlab_polynomials():
    """Load MATLAB reference data for polynomials."""
    from tests.conftest import load_matlab_ref
    return load_matlab_ref("polynomials.mat")
