"""Tests for chebfunjax.utils.transforms — Chebyshev/Legendre/Jacobi transforms.

JAX contract: jit=yes, vmap=yes (static n), grad=yes
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.transforms import (
    cheb2jac,
    cheb2leg,
    chebcoeffs2legcoeffs,
    chebvals2legcoeffs,
    coeffs2vals,
    jac2cheb,
    leg2cheb,
    legcoeffs2chebcoeffs,
    vals2coeffs,
)

# ===========================================================================
# Tier 1: Pure mathematical tests (no MATLAB)
# ===========================================================================


class TestVals2Coeffs:
    """Tests for vals2coeffs / coeffs2vals (DCT-I pair)."""

    def test_roundtrip_small(self):
        """vals2coeffs inverts coeffs2vals for small n."""
        for n in [2, 3, 5, 10]:
            c = jnp.array(np.random.default_rng(42).standard_normal(n), dtype=jnp.float64)
            v = coeffs2vals(c)
            c_back = vals2coeffs(v)
            npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-14, atol=1e-15)

    def test_roundtrip_medium(self):
        """Round-trip for n=50."""
        rng = np.random.default_rng(42)
        c = jnp.array(rng.standard_normal(50), dtype=jnp.float64)
        v = coeffs2vals(c)
        c_back = vals2coeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-13, atol=1e-14)

    def test_constant(self):
        """Constant function: c=[A,0,...,0] -> all values equal to A."""
        c = jnp.array([3.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.float64)
        v = coeffs2vals(c)
        npt.assert_allclose(np.array(v), 3.0, atol=1e-15)

    def test_linear(self):
        """T_1 = x: c=[0,1,0,...] -> values at Chebyshev points."""
        n = 5
        c = jnp.zeros(n, dtype=jnp.float64).at[1].set(1.0)
        v = coeffs2vals(c)
        # Values should be at ascending Chebyshev-2 points: cos(k*pi/(n-1)) reversed
        k = jnp.arange(n - 1, -1, -1, dtype=jnp.float64)
        x = jnp.cos(k * jnp.pi / (n - 1))
        npt.assert_allclose(np.array(v), np.array(x), atol=1e-15)

    def test_n1(self):
        """Scalar case."""
        c = jnp.array([2.5], dtype=jnp.float64)
        v = coeffs2vals(c)
        npt.assert_allclose(np.array(v), np.array(c), atol=1e-15)
        c_back = vals2coeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(c), atol=1e-15)

    def test_symmetry_even(self):
        """Even coefficients (c_{odd}=0) produce even/symmetric values."""
        c = jnp.array([1.0, 0.0, 0.5, 0.0, 0.2], dtype=jnp.float64)
        v = coeffs2vals(c)
        # Even function: v(x) = v(-x), so v[k] = v[n-1-k]
        npt.assert_allclose(np.array(v), np.array(v[::-1]), atol=1e-14)


class TestCheb2Leg:
    """Tests for cheb2leg — Chebyshev to Legendre coefficient conversion."""

    def test_T0_is_P0(self):
        """T_0 = 1 = P_0."""
        c_cheb = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
        c_leg = cheb2leg(c_cheb)
        expected = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_leg), np.array(expected), atol=1e-14)

    def test_T1_is_P1(self):
        """T_1 = x = P_1."""
        c_cheb = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        c_leg = cheb2leg(c_cheb)
        expected = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_leg), np.array(expected), atol=1e-14)

    def test_T2_to_legendre(self):
        """T_2 = 2x^2 - 1 = -1/3 P_0 + 4/3 P_2."""
        c_cheb = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
        c_leg = cheb2leg(c_cheb)
        expected = jnp.array([-1.0 / 3, 0.0, 4.0 / 3], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_leg), np.array(expected), atol=1e-14)

    def test_T3_to_legendre(self):
        """T_3 = 4x^3 - 3x = -3/5 P_1 + 8/5 P_3."""
        c_cheb = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
        c_leg = cheb2leg(c_cheb)
        expected = jnp.array([0.0, -3.0 / 5, 0.0, 8.0 / 5], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_leg), np.array(expected), atol=1e-14)

    def test_roundtrip(self):
        """cheb2leg then leg2cheb recovers input."""
        rng = np.random.default_rng(42)
        for n in [3, 5, 10, 30, 50]:
            c_cheb = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
            c_leg = cheb2leg(c_cheb)
            c_back = leg2cheb(c_leg)
            npt.assert_allclose(
                np.array(c_back), np.array(c_cheb),
                rtol=1e-12, atol=1e-14,
                err_msg=f"Round-trip failed for n={n}"
            )

    def test_n1(self):
        """Scalar: cheb2leg([c]) = [c]."""
        c = jnp.array([42.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(cheb2leg(c)), np.array(c), atol=1e-15)

    def test_normalize(self):
        """Normalized Legendre coefficients satisfy sum(c_norm^2 * (2k+1)) = integral(f^2)."""
        c_cheb = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        c_leg = cheb2leg(c_cheb)
        c_norm = cheb2leg(c_cheb, normalize=True)
        # c_norm = c_leg / sqrt(k + 1/2)
        norms = jnp.sqrt(jnp.arange(4, dtype=jnp.float64) + 0.5)
        npt.assert_allclose(np.array(c_norm), np.array(c_leg / norms), rtol=1e-14)


class TestLeg2Cheb:
    """Tests for leg2cheb — Legendre to Chebyshev coefficient conversion."""

    def test_P0_is_T0(self):
        """P_0 = 1 = T_0."""
        c_leg = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
        c_cheb = leg2cheb(c_leg)
        expected = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_cheb), np.array(expected), atol=1e-14)

    def test_P1_is_T1(self):
        """P_1 = x = T_1."""
        c_leg = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        c_cheb = leg2cheb(c_leg)
        expected = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_cheb), np.array(expected), atol=1e-14)

    def test_P2_to_chebyshev(self):
        """P_2 = (3x^2-1)/2 = 1/4*T_0 + 3/4*T_2."""
        c_leg = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
        c_cheb = leg2cheb(c_leg)
        expected = jnp.array([0.25, 0.0, 0.75], dtype=jnp.float64)
        npt.assert_allclose(np.array(c_cheb), np.array(expected), atol=1e-14)

    def test_roundtrip(self):
        """leg2cheb then cheb2leg recovers input."""
        rng = np.random.default_rng(42)
        for n in [3, 5, 10, 30, 50]:
            c_leg = jnp.array(rng.standard_normal(n), dtype=jnp.float64)
            c_cheb = leg2cheb(c_leg)
            c_back = cheb2leg(c_cheb)
            npt.assert_allclose(
                np.array(c_back), np.array(c_leg),
                rtol=1e-12, atol=1e-14,
                err_msg=f"Round-trip failed for n={n}"
            )

    def test_n1(self):
        """Scalar: leg2cheb([c]) = [c]."""
        c = jnp.array([42.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(leg2cheb(c)), np.array(c), atol=1e-15)

    def test_normalize(self):
        """With normalize=True, input is assumed orthonormal Legendre."""
        c_leg = jnp.array([1.0, 0.5, 0.3], dtype=jnp.float64)
        # Normalized input: multiply by sqrt(k+1/2) to get standard
        norms = jnp.sqrt(jnp.arange(3, dtype=jnp.float64) + 0.5)
        c_cheb_norm = leg2cheb(c_leg, normalize=True)
        c_cheb_manual = leg2cheb(c_leg * norms)
        npt.assert_allclose(np.array(c_cheb_norm), np.array(c_cheb_manual), rtol=1e-14)


class TestCheb2Jac:
    """Tests for cheb2jac / jac2cheb — Chebyshev to/from Jacobi."""

    def test_legendre_special_case(self):
        """cheb2jac(c, 0, 0) == cheb2leg(c)."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(cheb2jac(c, 0.0, 0.0)),
            np.array(cheb2leg(c)),
            rtol=1e-14
        )

    def test_chebyshev_special_case(self):
        """cheb2jac(c, -1/2, -1/2) scales T_n by known constants."""
        c = jnp.array([1.0, 0.5, 0.3], dtype=jnp.float64)
        c_jac = cheb2jac(c, -0.5, -0.5)
        # T_n = scl[n] * P_n^{(-1/2,-1/2)}
        # scl = [1, 1/2, 3/8]
        scl = np.array([1.0, 0.5, 0.5 * 1.5 / 2.0])
        npt.assert_allclose(np.array(c_jac), np.array(c) / scl, rtol=1e-14)

    def test_roundtrip(self):
        """cheb2jac then jac2cheb recovers input."""
        rng = np.random.default_rng(42)
        c = jnp.array(rng.standard_normal(20), dtype=jnp.float64)
        for a, b in [(0.5, 0.5), (1.0, 0.5), (2.0, 1.5), (0.1, 0.3), (3.0, 0.0)]:
            c_jac = cheb2jac(c, a, b)
            c_back = jac2cheb(c_jac, a, b)
            npt.assert_allclose(
                np.array(c_back), np.array(c),
                rtol=1e-12, atol=1e-14,
                err_msg=f"Round-trip failed for alpha={a}, beta={b}"
            )

    def test_T0_identity(self):
        """T_0 = 1 = P_0^(a,b) for any (a,b)."""
        for a, b in [(0.5, 0.5), (1.0, 2.0), (0.1, 0.3)]:
            c_cheb = jnp.array([1.0, 0.0, 0.0], dtype=jnp.float64)
            c_jac = cheb2jac(c_cheb, a, b)
            npt.assert_allclose(
                np.array(c_jac),
                np.array([1.0, 0.0, 0.0]),
                atol=1e-13,
                err_msg=f"T_0 identity failed for alpha={a}, beta={b}"
            )

    def test_n1(self):
        """Scalar case."""
        c = jnp.array([42.0], dtype=jnp.float64)
        npt.assert_allclose(np.array(cheb2jac(c, 1.0, 0.5)), np.array(c), atol=1e-15)
        npt.assert_allclose(np.array(jac2cheb(c, 1.0, 0.5)), np.array(c), atol=1e-15)


class TestJac2Cheb:
    """Tests for jac2cheb — Jacobi to Chebyshev coefficient conversion."""

    def test_legendre_special_case(self):
        """jac2cheb(c, 0, 0) == leg2cheb(c)."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(jac2cheb(c, 0.0, 0.0)),
            np.array(leg2cheb(c)),
            rtol=1e-14
        )

    def test_roundtrip(self):
        """jac2cheb then cheb2jac recovers input."""
        rng = np.random.default_rng(42)
        c = jnp.array(rng.standard_normal(20), dtype=jnp.float64)
        for a, b in [(0.5, 0.5), (1.0, 0.5), (2.0, 1.5)]:
            c_cheb = jac2cheb(c, a, b)
            c_back = cheb2jac(c_cheb, a, b)
            npt.assert_allclose(
                np.array(c_back), np.array(c),
                rtol=1e-12, atol=1e-14,
                err_msg=f"Round-trip failed for alpha={a}, beta={b}"
            )


class TestWrappers:
    """Tests for convenience wrapper functions."""

    def test_chebcoeffs2legcoeffs(self):
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(chebcoeffs2legcoeffs(c)),
            np.array(cheb2leg(c)),
            atol=1e-15
        )

    def test_legcoeffs2chebcoeffs(self):
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)
        npt.assert_allclose(
            np.array(legcoeffs2chebcoeffs(c)),
            np.array(leg2cheb(c)),
            atol=1e-15
        )

    def test_chebvals2legcoeffs_kind2(self):
        """chebvals2legcoeffs == cheb2leg(vals2coeffs(v))."""
        v = jnp.array([1.0, 0.5, 0.3, 0.7, 0.9], dtype=jnp.float64)
        c_leg = chebvals2legcoeffs(v, kind=2)
        c_expected = cheb2leg(vals2coeffs(v))
        npt.assert_allclose(np.array(c_leg), np.array(c_expected), atol=1e-14)

    def test_chebvals2legcoeffs_invalid_kind(self):
        v = jnp.array([1.0, 0.5], dtype=jnp.float64)
        with pytest.raises(ValueError, match="kind must be 1 or 2"):
            chebvals2legcoeffs(v, kind=3)


# ===========================================================================
# Tier 2: MATLAB cross-validation
# ===========================================================================


class TestCheb2LegMATLAB:
    """Compare cheb2leg against MATLAB Chebfun reference data."""

    @pytest.mark.matlab
    def test_cheb2leg_vs_matlab(self, matlab_transforms):
        for n in [8, 16, 32, 64]:
            c_in = matlab_transforms[f"cheb2leg_n{n}_in"]
            c_ref = matlab_transforms[f"cheb2leg_n{n}_out"]
            c_out = np.array(cheb2leg(jnp.array(c_in, dtype=jnp.float64)))
            npt.assert_allclose(
                c_out, c_ref,
                rtol=1e-12, atol=1e-14,
                err_msg=f"cheb2leg n={n}"
            )

    @pytest.mark.matlab
    def test_leg2cheb_vs_matlab(self, matlab_transforms):
        for n in [8, 16, 32, 64]:
            c_in = matlab_transforms[f"leg2cheb_n{n}_in"]
            c_ref = matlab_transforms[f"leg2cheb_n{n}_out"]
            c_out = np.array(leg2cheb(jnp.array(c_in, dtype=jnp.float64)))
            npt.assert_allclose(
                c_out, c_ref,
                rtol=1e-12, atol=1e-14,
                err_msg=f"leg2cheb n={n}"
            )


# ===========================================================================
# JIT / grad / vmap tests
# ===========================================================================


class TestJITCompatibility:
    """Verify transforms work under jax.jit."""

    def test_cheb2leg_jit(self):
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        jitted = jax.jit(cheb2leg)
        npt.assert_allclose(
            np.array(jitted(c)),
            np.array(cheb2leg(c)),
            rtol=1e-14
        )

    def test_leg2cheb_jit(self):
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        jitted = jax.jit(leg2cheb)
        npt.assert_allclose(
            np.array(jitted(c)),
            np.array(leg2cheb(c)),
            rtol=1e-14
        )

    def test_vals2coeffs_jit(self):
        v = jnp.array([1.0, 0.5, 0.3, 0.7, 0.9], dtype=jnp.float64)
        jitted = jax.jit(vals2coeffs)
        npt.assert_allclose(
            np.array(jitted(v)),
            np.array(vals2coeffs(v)),
            rtol=1e-14
        )

    def test_coeffs2vals_jit(self):
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        jitted = jax.jit(coeffs2vals)
        npt.assert_allclose(
            np.array(jitted(c)),
            np.array(coeffs2vals(c)),
            rtol=1e-14
        )

    def test_cheb2jac_jit(self):
        """cheb2jac under JIT (with static alpha, beta)."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        jitted = jax.jit(functools.partial(cheb2jac, alpha=0.5, beta=0.5))
        npt.assert_allclose(
            np.array(jitted(c)),
            np.array(cheb2jac(c, 0.5, 0.5)),
            rtol=1e-14
        )

    def test_jac2cheb_jit(self):
        """jac2cheb under JIT (with static alpha, beta)."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)
        jitted = jax.jit(functools.partial(jac2cheb, alpha=0.5, beta=0.5))
        npt.assert_allclose(
            np.array(jitted(c)),
            np.array(jac2cheb(c, 0.5, 0.5)),
            rtol=1e-14
        )


class TestGradCompatibility:
    """Verify transforms are differentiable."""

    def test_cheb2leg_grad(self):
        """Gradient of sum(cheb2leg(c)) w.r.t. c."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)

        def f(c):
            return jnp.sum(cheb2leg(c))

        g = jax.grad(f)(c)
        # Check that grad returns a valid array of the same shape
        assert g.shape == c.shape
        assert jnp.all(jnp.isfinite(g))

    def test_leg2cheb_grad(self):
        """Gradient of sum(leg2cheb(c)) w.r.t. c."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)

        def f(c):
            return jnp.sum(leg2cheb(c))

        g = jax.grad(f)(c)
        assert g.shape == c.shape
        assert jnp.all(jnp.isfinite(g))

    def test_vals2coeffs_grad(self):
        """Gradient of sum(vals2coeffs(v)) w.r.t. v."""
        v = jnp.array([1.0, 0.5, 0.3, 0.7, 0.9], dtype=jnp.float64)

        def f(v):
            return jnp.sum(vals2coeffs(v))

        g = jax.grad(f)(v)
        assert g.shape == v.shape
        assert jnp.all(jnp.isfinite(g))

    def test_roundtrip_grad(self):
        """Gradient through cheb2leg + leg2cheb round-trip should be identity-like."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1, 0.05], dtype=jnp.float64)

        def f(c):
            return jnp.sum(leg2cheb(cheb2leg(c)))

        g = jax.grad(f)(c)
        # Derivative of sum(c) w.r.t. c is all-ones if roundtrip is exact
        npt.assert_allclose(np.array(g), 1.0, rtol=1e-12)

    def test_cheb2leg_grad_numerical(self):
        """Numerical gradient check for cheb2leg."""
        c = jnp.array([1.0, 0.5, 0.3, 0.1], dtype=jnp.float64)

        def f(c):
            return jnp.sum(cheb2leg(c) ** 2)

        g_analytic = jax.grad(f)(c)

        # Numerical gradient
        eps = 1e-7
        g_numerical = np.zeros_like(c)
        for i in range(len(c)):
            c_plus = c.at[i].add(eps)
            c_minus = c.at[i].add(-eps)
            g_numerical[i] = (float(f(c_plus)) - float(f(c_minus))) / (2 * eps)

        npt.assert_allclose(np.array(g_analytic), g_numerical, rtol=1e-5)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def matlab_transforms():
    """Load MATLAB reference data for transforms."""
    from tests.conftest import load_matlab_ref
    return load_matlab_ref("transforms.mat")
