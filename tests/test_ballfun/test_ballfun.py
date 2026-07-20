"""Tests for Ballfun — Chebyshev-Fourier-Fourier approximation on the unit ball.

JAX contract:
    construction   : jit=NO (adaptive Python loop)
    evaluation     : jit=YES (via fevalm / __call__)

Test coverage (Tier 1 — unit tests, no MATLAB required):
    - Constant function: integral == 4*pi/3
    - Arithmetic: negation, scalar addition, scalar multiplication
    - Coefficient round-trip: from_coeffs / fevalm consistency
    - repr includes shape information
    - from_function with fixed_size
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.ballfun.ballfun import Ballfun, _coeffs2vals_3d, _vals2coeffs_3d

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL = 1e-8
RTOL = 1e-10


# ===========================================================================
# Spectral transform tests (independent of constructor)
# ===========================================================================


class TestSpectralTransforms:
    """Round-trip tests for vals2coeffs and coeffs2vals."""

    def test_round_trip_real(self):
        """vals -> coeffs -> vals should be identity."""
        rng = np.random.default_rng(42)
        vals = rng.standard_normal((5, 4, 4))
        cfs = _vals2coeffs_3d(vals)
        vals_back = _coeffs2vals_3d(cfs)
        npt.assert_allclose(np.real(vals_back), vals, atol=1e-12, rtol=0)

    def test_round_trip_complex(self):
        """Complex vals -> coeffs -> vals round-trip."""
        rng = np.random.default_rng(7)
        vals = rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4))
        cfs = _vals2coeffs_3d(vals)
        vals_back = _coeffs2vals_3d(cfs)
        npt.assert_allclose(vals_back, vals, atol=1e-12, rtol=0)


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Ballfun.from_function and from_coeffs."""

    def test_fixed_size_constant(self):
        """from_function with fixed_size for a constant function."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x) * 2.0,
            fixed_size=(5, 4, 4),
        )
        assert f.shape[0] % 2 == 1, "m should be odd"
        assert f.shape[1] % 2 == 0, "n should be even"
        assert f.shape[2] % 2 == 0, "p should be even"
        assert f.is_real

    def test_fixed_size_spherical(self):
        """from_function in spherical coords with fixed_size."""
        f = Ballfun.from_function(
            lambda r, lam, th: r**2,
            spherical=True,
            fixed_size=(5, 4, 4),
        )
        assert f.shape[0] >= 3

    def test_from_coeffs(self):
        """Construct from a coefficient tensor."""
        m, n, p = 5, 4, 4
        cfs = jnp.zeros((m, n, p), dtype=jnp.complex128)
        # Set DC coefficient to 1.0
        cfs = cfs.at[0, n // 2, p // 2].set(1.0 + 0j)
        f = Ballfun.from_coeffs(cfs)
        assert f.shape == (m, n, p)

    def test_repr(self):
        """repr includes shape."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x),
            fixed_size=(5, 4, 4),
        )
        s = repr(f)
        assert "Ballfun" in s
        assert "shape" in s


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Ballfun evaluation."""

    def test_constant_fevalm(self):
        """Constant function evaluates to that constant everywhere."""
        const = 3.14
        f = Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, const),
            fixed_size=(5, 4, 4),
        )
        r_pts = jnp.array([0.0, 0.3, 0.7, 1.0])
        lam_pts = jnp.array([0.0, 1.0, -1.0])
        th_pts = jnp.array([0.5, 1.0, 2.0])
        vals = f.fevalm(r_pts, lam_pts, th_pts)
        assert vals.shape == (4, 3, 3)
        npt.assert_allclose(np.real(np.array(vals)), const, atol=1e-6, rtol=0)

    def test_r2_fevalm(self):
        """f = r^2 evaluated on a grid matches r^2."""
        f = Ballfun.from_function(
            lambda r, lam, th: r**2,
            spherical=True,
            fixed_size=(9, 4, 4),
        )
        r_pts = jnp.array([0.0, 0.5, 1.0])
        lam_pts = jnp.array([0.0])
        th_pts = jnp.array([jnp.pi / 2])
        vals = np.real(np.array(f.fevalm(r_pts, lam_pts, th_pts)))
        expected = np.array([0.0, 0.25, 1.0])
        npt.assert_allclose(vals[:, 0, 0], expected, atol=1e-4, rtol=0)


# ===========================================================================
# Integral tests
# ===========================================================================


class TestIntegral:
    """Tests for Ballfun.sum() / integral()."""

    def test_constant_one_integral(self):
        """Integral of 1 over unit ball = 4*pi/3."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.ones_like(x),
            fixed_size=(5, 4, 4),
        )
        I = f.integral()
        expected = 4.0 * float(np.pi) / 3.0
        npt.assert_allclose(I, expected, atol=1e-4, rtol=1e-4)

    def test_constant_two_integral(self):
        """Integral of 2 over unit ball = 8*pi/3."""
        f = Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, 2.0),
            fixed_size=(5, 4, 4),
        )
        I = f.integral()
        expected = 8.0 * float(np.pi) / 3.0
        npt.assert_allclose(I, expected, atol=1e-4, rtol=1e-4)


# ===========================================================================
# Arithmetic tests
# ===========================================================================


class TestArithmetic:
    """Tests for Ballfun arithmetic operations."""

    def _const_ball(self, c: float) -> Ballfun:
        return Ballfun.from_function(
            lambda x, y, z: jnp.full_like(x, c),
            fixed_size=(5, 4, 4),
        )

    def test_negation(self):
        """Negation: integral of -f = -integral(f)."""
        f = self._const_ball(2.0)
        neg_f = -f
        I_f = f.integral()
        I_neg = neg_f.integral()
        npt.assert_allclose(I_neg, -I_f, atol=1e-6, rtol=0)

    def test_scalar_multiply(self):
        """Scalar multiply: integral of 3*f = 3*integral(f)."""
        f = self._const_ball(1.0)
        g = 3.0 * f
        I_f = f.integral()
        I_g = g.integral()
        npt.assert_allclose(I_g, 3.0 * I_f, atol=1e-6, rtol=0)

    def test_scalar_add(self):
        """Adding scalar c shifts integral by c * 4*pi/3."""
        f = self._const_ball(1.0)
        g = f + 2.0
        I_f = f.integral()
        I_g = g.integral()
        vol = 4.0 * float(np.pi) / 3.0
        npt.assert_allclose(I_g, I_f + 2.0 * vol, atol=1e-4, rtol=0)


class TestBallfunPoisson:
    """Spectral Poisson solver on the ball (Opus 4.8, task #17).

    Verified against manufactured solutions u = (1-r^2) r^l Y_l^m, whose
    Laplacian is -(4l+6) r^l Y_l^m.
    """

    _pts = [(0.3, 0.4, 0.5), (0.6, 1.0, 2.0), (0.8, -1.5, 1.2)]

    def test_manufactured_modes(self):
        import jax.numpy as jnp
        import numpy as np

        from chebfunjax.ballfun.ballfun import Ballfun
        from chebfunjax.spherefun.spherefun import _real_ylm_values
        for l, m in [(0, 0), (1, 0), (2, 1), (3, -2)]:
            def rhs(r, lam, th, _l=l, _m=m):
                return -(4 * _l + 6) * jnp.asarray(r) ** _l \
                    * _real_ylm_values(_l, _m, jnp.asarray(lam),
                                       jnp.asarray(th))
            u = Ballfun.poisson(rhs, lmax=max(4, l + 1))
            got = np.array([float(u(jnp.array(r), jnp.array(la),
                                    jnp.array(t)))
                            for r, la, t in self._pts])
            want = np.array(
                [(1 - r ** 2) * r ** l
                 * float(_real_ylm_values(l, m, jnp.array(la), jnp.array(t)))
                 for r, la, t in self._pts])
            np.testing.assert_allclose(got, want, atol=1e-10)


class TestBallfunComplexPartsAndComposition:
    """Core exercise of the Fable 5 additions: real/imag/conj/abs, the
    iszero/isequal predicates, and the log/tan/tanh/sinh/cosh composition
    ops.  Mirrors the MATLAB-port assertions so these paths are covered."""

    def test_complex_parts_and_abs(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda x, y, z: x + 1j * y)
        assert (f.real() - Ballfun.from_function(
            lambda x, y, z: x)).norm() < 1e4 * np.finfo(np.float64).eps
        g = Ballfun.from_function(lambda x, y, z: x + 1j * y * z)
        assert (g.imag() - Ballfun.from_function(
            lambda x, y, z: y * z)).norm() < 1e4 * np.finfo(np.float64).eps
        assert (f.conj() - Ballfun.from_function(
            lambda x, y, z: x - 1j * y)).norm() \
            < 1e2 * np.finfo(np.float64).eps
        a = abs(Ballfun.from_function(lambda x, y, z: 1j * x ** 2))
        assert (a - Ballfun.from_function(
            lambda x, y, z: x ** 2)).norm() \
            < 1e2 * np.finfo(np.float64).eps

    def test_predicates(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda x, y, z: 1.0 + 0.0 * x)
        assert (f - f).iszero()
        assert not Ballfun.from_function(
            lambda x, y, z: 1e-20 + 0.0 * x).iszero()
        assert f.isequal(f + f - f)

    def test_composition_ops(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        tol = 1e4 * np.finfo(np.float64).eps
        f = Ballfun.from_function(lambda x, y, z: jnp.exp(y))
        assert (f.log() - Ballfun.from_function(
            lambda x, y, z: y)).norm() < tol
        s = Ballfun.from_function(lambda x, y, z: jnp.sin(y))
        assert (s.tan() - Ballfun.from_function(
            lambda x, y, z: jnp.tan(jnp.sin(y)))).norm() < tol
        b = Ballfun.from_function(lambda x, y, z: y)
        assert (b.sinh() - Ballfun.from_function(
            lambda x, y, z: jnp.sinh(y))).norm() < tol
        assert (b.cosh() - Ballfun.from_function(
            lambda x, y, z: jnp.cosh(y))).norm() < 1e2 * np.finfo(
            np.float64).eps


# ===========================================================================
# Partial integration: sum(dim) and sum2(dims) (Fable 5)
# ===========================================================================


class TestBallfunPartialIntegration:
    """Core mirror tests for Ballfun.sum(dim) and Ballfun.sum2(dims)."""

    @staticmethod
    def _bf(fn):
        from chebfunjax.ballfun.ballfun import Ballfun
        return Ballfun.from_function(fn, spherical=True)

    def test_full_sum_still_scalar(self):
        f = self._bf(lambda r, lam, th: jnp.ones_like(r))
        val = f.sum()
        assert isinstance(val, float)
        npt.assert_allclose(val, 4.0 * np.pi / 3.0, atol=1e-11)

    def test_sum_dim1_returns_spherefun(self):
        from chebfunjax.spherefun.spherefun import Spherefun
        f = self._bf(lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th))
        g = f.sum(1)
        assert isinstance(g, Spherefun)
        lam = np.linspace(-np.pi, np.pi, 15)
        th = np.linspace(0.05, np.pi - 0.05, 11)
        LL, TT = np.meshgrid(lam, th)
        got = np.asarray(g(jnp.asarray(LL), jnp.asarray(TT)))
        assert np.max(np.abs(got - np.cos(LL) * np.sin(TT) / 4.0)) < 1e-11

    def test_sum_dim2_and_dim3_return_diskfun(self):
        from chebfunjax.diskfun.diskfun import Diskfun
        f = self._bf(lambda r, lam, th: (r * jnp.sin(lam) * jnp.sin(th)) ** 2)
        g2 = f.sum(2)
        assert isinstance(g2, Diskfun)
        th = np.linspace(-np.pi, np.pi, 15)
        r = np.linspace(0.1, 0.9, 11)
        TH, R = np.meshgrid(th, r)
        got = np.asarray(g2(jnp.asarray(TH), jnp.asarray(R)))
        assert np.max(np.abs(got - np.pi * R**2 * np.sin(TH) ** 2)) < 1e-11
        f3 = self._bf(lambda r, lam, th: r * jnp.cos(lam) * jnp.sin(th))
        g3 = f3.sum(3)
        assert isinstance(g3, Diskfun)
        got3 = np.asarray(g3(jnp.asarray(TH), jnp.asarray(R)))
        assert np.max(np.abs(got3 - R * np.cos(TH) * np.pi / 2.0)) < 1e-11

    def test_sum_invalid_dim(self):
        import pytest
        f = self._bf(lambda r, lam, th: jnp.ones_like(r))
        with pytest.raises(ValueError):
            f.sum(4)

    def test_sum2_defaults_and_variants(self):
        from chebfunjax.chebfun1d.chebfun import Chebfun
        # default dims=(2,3): constant 4pi in r.
        f = self._bf(lambda r, lam, th: jnp.ones_like(r))
        g = f.sum2()
        assert isinstance(g, Chebfun)
        r = np.linspace(0.0, 1.0, 11)
        assert np.max(np.abs(np.asarray(g(jnp.asarray(r))) - 4.0 * np.pi)) < 1e-11
        # dims=(1,3): trig in lambda, cos(lam) pi/8.
        f = self._bf(lambda r, lam, th: r * jnp.sin(th) * jnp.cos(lam))
        g = f.sum2((1, 3))
        lam = np.linspace(-np.pi, np.pi, 17)
        assert np.max(np.abs(np.asarray(g(jnp.asarray(lam)))
                             - np.cos(lam) * np.pi / 8.0)) < 1e-11
        # dims=(1,2): trig in theta; integrand odd in lambda -> zero here.
        g = f.sum2((1, 2))
        th = np.linspace(-np.pi, np.pi, 17)
        assert np.max(np.abs(np.asarray(g(jnp.asarray(th))))) < 1e-11


# ===========================================================================
# Slice extraction: to_spherefun(r) and to_diskfun(axis, c) (Fable 5)
# ===========================================================================


class TestBallfunSliceExtraction:
    """Core mirror tests for Ballfun.to_spherefun and Ballfun.to_diskfun."""

    def test_to_spherefun_shell(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        from chebfunjax.spherefun.spherefun import Spherefun
        f = Ballfun.from_function(lambda x, y, z: jnp.sin(z))
        g = f.to_spherefun(0.5)
        assert isinstance(g, Spherefun)
        lam = np.linspace(-np.pi, np.pi, 20, endpoint=False)
        th = np.linspace(0.05, np.pi - 0.05, 15)
        LL, TT = np.meshgrid(lam, th)
        got = np.asarray(g(jnp.asarray(LL), jnp.asarray(TT)))
        assert np.max(np.abs(got - np.sin(0.5 * np.cos(TT)))) < 1e-12

    def test_to_diskfun_equatorial(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        from chebfunjax.diskfun.diskfun import Diskfun
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        g = f.to_diskfun()
        assert isinstance(g, Diskfun)
        th = np.linspace(-np.pi, np.pi, 20)
        r = np.linspace(0.05, 0.9, 15)
        TH, R = np.meshgrid(th, r)
        x = R * np.cos(TH)
        y = R * np.sin(TH)
        got = np.asarray(g(jnp.asarray(TH), jnp.asarray(R)))
        assert np.max(np.abs(got - np.cos(x * y))) < 1e-12

    def test_to_diskfun_x_axis(self):
        from chebfunjax.ballfun.ballfun import Ballfun
        # f = y z; slice x=0 maps disk (x,y) -> ball (0, x, z=y) -> x y.
        f = Ballfun.from_function(lambda x, y, z: y * z)
        g = f.to_diskfun("x")
        th = np.linspace(-np.pi, np.pi, 20)
        r = np.linspace(0.05, 0.9, 15)
        TH, R = np.meshgrid(th, r)
        a = R * np.cos(TH)
        b = R * np.sin(TH)
        got = np.asarray(g(jnp.asarray(TH), jnp.asarray(R)))
        assert np.max(np.abs(got - a * b)) < 1e-12

    def test_to_diskfun_bad_axis_and_offset(self):
        import pytest

        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda x, y, z: x)
        with pytest.raises(ValueError):
            f.to_diskfun("w")
        with pytest.raises(ValueError):
            f.to_diskfun("z", 1.5)
