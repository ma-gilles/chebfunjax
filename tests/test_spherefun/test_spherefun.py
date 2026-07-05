"""Tests for Spherefun — low-rank function approximation on the unit sphere.

JAX contract:
    construction   : jit=NO (Python adaptive loop)
    evaluation     : jit=YES, vmap=YES, grad=YES

Test coverage:
    - Constant function integral = 4*pi
    - Spherical harmonic orthogonality
    - JIT evaluation
    - vmap evaluation
    - Construction and evaluation accuracy
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.spherefun.spherefun import Spherefun

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RTOL = 1e-10
ATOL = 1e-8
# Looser tolerance for functions whose doubled-up polar extension is only
# piecewise smooth (e.g. cos(theta)*cos(lam), where the doubled column for
# the "minus" part has a derivative kink at theta=0 and theta=pi):
RTOL_UNSMOOTH = 1e-6


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Spherefun.from_function construction."""

    def test_constant_rank(self):
        """A constant function has rank >= 1."""
        f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam) * 3.14)
        assert f.rank >= 1, f"Constant should have rank >= 1, got {f.rank}"

    def test_cos_theta_rank(self):
        """cos(theta) should be low rank (single spherical harmonic Y_1^0)."""
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        assert f.rank >= 1
        assert f.rank <= 8, f"cos(theta) rank {f.rank} unexpectedly high"

    def test_sin_theta_cos_lam(self):
        """sin(theta)*cos(lam) (Y_1^1 harmonic) should be low rank."""
        f = Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        assert f.rank >= 1
        assert f.rank <= 8, f"sin(th)*cos(lam) rank {f.rank} unexpectedly high"

    def test_repr(self):
        """repr includes rank information."""
        f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam))
        s = repr(f)
        assert "Spherefun" in s
        assert "rank" in s

    def test_zero_function(self):
        """Zero function is handled without error."""
        f = Spherefun.from_function(lambda lam, th: jnp.zeros_like(lam))
        assert f.rank >= 1


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Spherefun.__call__ evaluation."""

    def test_constant_eval(self):
        """Constant function evaluates correctly."""
        val = 2.718
        f = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, val))
        lam_test = jnp.array([0.0, 1.0, -0.5, 2.0])
        th_test = jnp.array([0.5, 1.0, np.pi / 4, 2.0])
        y = f(lam_test, th_test)
        npt.assert_allclose(y, val, rtol=RTOL, atol=ATOL)

    def test_cos_theta_eval(self):
        """cos(theta) evaluates correctly at various spherical points."""
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        lam_test = jnp.array([0.0, 1.0, -np.pi / 3, np.pi / 2])
        th_test = jnp.array([0.0, np.pi / 4, np.pi / 2, np.pi])
        y = f(lam_test, th_test)
        expected = jnp.cos(th_test)
        npt.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)

    def test_sin_theta_cos_lam_eval(self):
        """sin(theta)*cos(lam) evaluates correctly."""
        f = Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        lam_test = jnp.array([0.0, np.pi / 4, -np.pi / 3, np.pi / 6])
        th_test = jnp.array([np.pi / 4, np.pi / 2, 2.0, np.pi / 3])
        y = f(lam_test, th_test)
        expected = jnp.sin(th_test) * jnp.cos(lam_test)
        npt.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)

    def test_pole_eval(self):
        """Evaluation at the poles (theta=0 and theta=pi) works."""
        # At north pole theta=0: f = cos(0) = 1
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        lam_test = jnp.array([0.0, 1.0, -0.5])
        # North pole
        th_north = jnp.zeros(3)
        y_north = f(lam_test, th_north)
        npt.assert_allclose(y_north, 1.0, rtol=RTOL, atol=ATOL)
        # South pole
        th_south = jnp.full(3, np.pi)
        y_south = f(lam_test, th_south)
        npt.assert_allclose(y_south, -1.0, rtol=RTOL, atol=ATOL)

    def test_scalar_eval(self):
        """Scalar evaluation works.

        Note: cos(theta)*cos(lam) lives entirely in the BMC-I "minus" part
        of the decomposition.  Its doubled-up column is an odd extension of
        cos(theta) which has derivative kinks at theta=0 and theta=pi, so
        the Fourier series converges only as O(1/k^2) and we use a loose
        tolerance.
        """
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th) * jnp.cos(lam))
        y = f(jnp.array(0.5), jnp.array(1.0))
        expected = np.cos(1.0) * np.cos(0.5)
        npt.assert_allclose(float(y), expected, rtol=RTOL_UNSMOOTH, atol=RTOL_UNSMOOTH)

    def test_jit_eval(self):
        """Evaluation is JIT-compilable."""
        f = Spherefun.from_function(lambda lam, th: jnp.sin(th) * jnp.cos(lam))
        jit_f = eqx.filter_jit(f)
        lam_test = jnp.linspace(-np.pi, np.pi, 10)
        th_test = jnp.linspace(0.0, np.pi, 10)
        y_jit = jit_f(lam_test, th_test)
        y_ref = f(lam_test, th_test)
        npt.assert_allclose(y_jit, y_ref, rtol=1e-12)

    def test_vmap_eval(self):
        """Evaluation is vmap-compatible."""
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        lam_batch = jnp.linspace(-np.pi, np.pi, 20)
        th_batch = jnp.linspace(0.0, np.pi, 20)
        y = jax.vmap(lambda lm, t: f(lm, t))(lam_batch, th_batch)
        y_ref = f(lam_batch, th_batch)
        npt.assert_allclose(y, y_ref, rtol=1e-12)


# ===========================================================================
# Integration tests
# ===========================================================================


class TestIntegration:
    """Tests for Spherefun.sum() — integral over the unit sphere."""

    def test_constant_integral_4pi(self):
        """Integral of 1 over the unit sphere equals 4*pi.

        ∫∫_S 1 * sin(theta) d(theta) d(lam)
            = ∫_{-pi}^{pi} d(lam) * ∫_0^pi sin(theta) d(theta)
            = 2*pi * 2 = 4*pi.
        """
        f = Spherefun.from_function(lambda lam, th: jnp.ones_like(lam))
        integral = f.sum()
        npt.assert_allclose(float(integral), 4.0 * np.pi, rtol=1e-8, atol=1e-8)

    def test_constant_c_integral(self):
        """Integral of constant c equals c * 4*pi."""
        c = 2.5
        f = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, c))
        integral = f.sum()
        npt.assert_allclose(float(integral), c * 4.0 * np.pi, rtol=1e-8, atol=1e-8)

    def test_spherical_harmonic_Y00_integral(self):
        """Y_0^0 = 1/sqrt(4*pi); integral over sphere = sqrt(4*pi).

        ∫∫_S (1/sqrt(4*pi)) sin(theta) d(theta) d(lam) = 4*pi / sqrt(4*pi) = sqrt(4*pi).
        """
        norm = 1.0 / np.sqrt(4.0 * np.pi)
        f = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, norm))
        integral = f.sum()
        expected = np.sqrt(4.0 * np.pi)
        npt.assert_allclose(float(integral), expected, rtol=1e-8, atol=1e-8)

    def test_cos_theta_integral(self):
        """Integral of cos(theta) over sphere is zero by symmetry.

        ∫∫_S cos(theta) sin(theta) d(theta) d(lam) = 0.
        (Integral of x over the sphere in Cartesian form = 0 by symmetry.)
        """
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        integral = f.sum()
        npt.assert_allclose(float(integral), 0.0, atol=1e-8)

    def test_spherical_harmonic_orthogonality(self):
        """Orthogonality of spherical harmonics: ∫∫_S Y_l^m * Y_l'^m' dS = delta.

        Y_1^0(theta, lam) = sqrt(3/(4*pi)) * cos(theta)
        Y_1^1(theta, lam) = -sqrt(3/(8*pi)) * sin(theta) * cos(lam) (Condon-Shortley)

        Orthogonality: ∫∫ Y_1^0 * Y_1^0 dS = 1.
        """
        # Y_1^0 = sqrt(3/(4*pi)) * cos(theta)
        norm_10 = np.sqrt(3.0 / (4.0 * np.pi))
        Spherefun.from_function(lambda lam, th: jnp.full_like(lam, norm_10) * jnp.cos(th))
        # ∫∫ (Y_1^0)^2 dS should equal 1
        g = Spherefun.from_function(lambda lam, th: (norm_10 * jnp.cos(th)) ** 2)
        integral = g.sum()
        npt.assert_allclose(float(integral), 1.0, rtol=1e-8, atol=1e-7)

    def test_Y10_times_1_is_zero(self):
        """∫∫_S Y_1^0 * 1 dS = 0 (Y_1^0 orthogonal to Y_0^0 constant part's norm).

        Actually ∫∫ Y_1^0 dS = sqrt(3/(4*pi)) * ∫∫ cos(theta) sin(theta) dth dlam = 0.
        """
        norm_10 = np.sqrt(3.0 / (4.0 * np.pi))
        f = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, norm_10) * jnp.cos(th))
        integral = f.sum()
        npt.assert_allclose(float(integral), 0.0, atol=1e-8)

    def test_sin2_theta_integral(self):
        """Integral of sin^2(theta) over sphere.

        ∫∫ sin^2(theta) sin(theta) dth dlam
            = 2*pi * ∫_0^pi sin^3(theta) d(theta)
            = 2*pi * 4/3 = 8*pi/3.
        """
        f = Spherefun.from_function(lambda lam, th: jnp.sin(th) ** 2)
        integral = f.sum()
        expected = 8.0 * np.pi / 3.0
        npt.assert_allclose(float(integral), expected, rtol=1e-7, atol=1e-7)


class TestSphericalCalculus:
    """Spectral Laplacian / Poisson on the sphere (Claude Opus 4.8).

    These use the exact harmonic identities as ground truth, so they
    verify correctness independently of the implementation:
      Delta Y_l^m = -l(l+1) Y_l^m,  and  poisson(laplacian(u)) == u.
    """

    _grid = (jnp.linspace(-3.0, 3.0, 11), jnp.linspace(0.15, 2.98, 11))

    def _sample(self, f):
        L, T = jnp.meshgrid(*self._grid)
        return np.asarray(f(L.ravel(), T.ravel()))

    def test_sphharm_matches_scipy(self):
        from scipy.special import sph_harm_y
        L, T = jnp.meshgrid(*self._grid)
        lam = np.asarray(L.ravel())
        th = np.asarray(T.ravel())
        for l, m in [(2, 0), (4, 2), (5, -3)]:
            got = np.asarray(Spherefun.sphharm(l, m)(L.ravel(), T.ravel()))
            if m == 0:
                sc = np.real(sph_harm_y(l, 0, th, lam))
            elif m > 0:
                sc = np.sqrt(2) * (-1) ** m * np.real(
                    sph_harm_y(l, m, th, lam))
            else:
                sc = np.sqrt(2) * (-1) ** abs(m) * np.imag(
                    sph_harm_y(l, abs(m), th, lam))
            mask = np.abs(sc) > 1e-3
            # equal up to a global sign convention
            ratio = got[mask] / sc[mask]
            assert np.std(ratio) < 1e-10
            assert abs(abs(np.mean(ratio)) - 1.0) < 1e-8

    def test_laplacian_eigenvalue_identity(self):
        """Delta Y_l^m = -l(l+1) Y_l^m to machine precision."""
        for l, m in [(1, 0), (2, 1), (3, -2), (4, 2), (6, 3)]:
            Y = Spherefun.sphharm(l, m)
            got = self._sample(Y.laplacian())
            want = -l * (l + 1) * self._sample(Y)
            npt.assert_allclose(got, want, atol=1e-11)

    def test_laplacian_linear_combination(self):
        Y31 = Spherefun.sphharm(3, 1)
        Y42 = Spherefun.sphharm(4, -2)
        f = Spherefun.from_function(
            lambda lam, th: Y31(lam, th) + 0.5 * Y42(lam, th))
        got = self._sample(f.laplacian())
        want = -12 * self._sample(Y31) - 0.5 * 20 * self._sample(Y42)
        npt.assert_allclose(got, want, atol=1e-10)

    def test_poisson_round_trip(self):
        """poisson(laplacian(u), const=0) recovers a zero-mean u."""
        Y31 = Spherefun.sphharm(3, 1)
        Y42 = Spherefun.sphharm(4, -2)
        u = Spherefun.from_function(
            lambda lam, th: Y31(lam, th) + 0.5 * Y42(lam, th))
        usol = Spherefun.poisson(u.laplacian(), const=0.0)
        npt.assert_allclose(self._sample(usol), self._sample(u),
                            atol=1e-11)

    def test_poisson_matlab_reference(self):
        """The example from MATLAB @spherefun/poisson.m docstring."""
        def f(lam, th):
            lam = jnp.asarray(lam)
            th = jnp.asarray(th)
            return -6 * (-1 + 5 * jnp.cos(2 * th)) * jnp.sin(lam) \
                * jnp.sin(2 * th)

        def exact(lam, th):
            lam = np.asarray(lam)
            th = np.asarray(th)
            return (-2 * np.sin(lam) * np.sin(2 * th) * np.sin(th) ** 2
                    - np.sin(lam) * np.sin(th) * np.cos(th)
                    + 0.5 * np.sin(lam) * np.sin(2 * th) * np.cos(2 * th))

        usol = Spherefun.poisson(f, const=0.0, lmax=8)
        L, T = jnp.meshgrid(*self._grid)
        got = np.asarray(usol(L.ravel(), T.ravel()))
        want = exact(np.asarray(L.ravel()), np.asarray(T.ravel()))
        # match up to the additive mean (const was set to 0, not the
        # exact solution's mean)
        npt.assert_allclose(got - got.mean(), want - want.mean(),
                            atol=1e-10)

    def test_mean(self):
        """mean() of a nonzero-degree harmonic is ~0; of a constant is it."""
        Y42 = Spherefun.sphharm(4, 2)
        assert abs(float(Y42.mean())) < 1e-8
        c = Spherefun.from_function(lambda lam, th: jnp.full_like(lam, 2.5))
        npt.assert_allclose(float(c.mean()), 2.5, atol=1e-8)


class TestConstructorMixedOrderBug:
    """Regression test for a pre-existing Spherefun.from_function bug.

    Discovered by Claude Opus 4.8 while validating a spectral gradient:
    ``from_function`` mis-reconstructs a sum of spherical harmonics whose
    sine orders differ (e.g. |m| = 1 and |m| = 3) at certain degrees --
    the BMC Gaussian-elimination pivoting does not resolve the rank-2
    longitude structure. Single harmonics, same-order sums, and cosine
    sums all reconstruct correctly, so the bug has a narrow trigger.

    This affects EVERY operation on such a spherefun (evaluation, sum,
    laplacian, ...), not just the constructor. It is NOT introduced by
    the laplacian/poisson work -- those are correct wherever
    from_function reconstructs the input faithfully.

    Marked xfail so the suite tracks it and flags when the constructor
    is fixed. See HANDOFF.md task "from_function mixed-order".
    """

    @pytest.mark.xfail(reason="pre-existing BMC constructor bug: mixed "
                              "sine-order harmonics mis-reconstruct",
                       strict=True)
    def test_mixed_sine_order_reconstructs(self):
        L, T = jnp.meshgrid(jnp.linspace(-3.0, 3.0, 9),
                            jnp.linspace(0.25, 2.85, 9))

        def target(lam, theta):
            return (Spherefun.sphharm(2, -1)(lam, theta)
                    + Spherefun.sphharm(4, -3)(lam, theta))

        f = Spherefun.from_function(target)
        got = np.asarray(f(L.ravel(), T.ravel()))
        want = np.asarray(target(L.ravel(), T.ravel()))
        npt.assert_allclose(got, want, atol=1e-8)
