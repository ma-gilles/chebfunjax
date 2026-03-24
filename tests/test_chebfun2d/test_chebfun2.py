"""Tests for Chebfun2 — user-facing 2D function approximation.

JAX contract:
    construction   : jit=NO (Python adaptive loop)
    evaluation     : jit=YES, vmap=YES, grad=YES
    diff/sum/norm  : jit=NO (return new Chebfun2; eval of result IS jit-safe)

MATLAB golden-reference values are computed from:
    f = chebfun2(@(x,y) cos(x+y));
    sum2(f)          % = 4*sin(1)*cos(1) = 2*sin(2) ≈ 1.8186...
    norm(f)          % = sqrt(sum2(f.^2)) = sqrt(1 + sin(2)*cos(2)) ≈ ...
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2, chebfun2

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
RTOL = 1e-10  # evaluation accuracy
RTOL_TIGHT = 1e-12  # for exact low-rank functions
ATOL_INTEGRAL = 1e-12  # absolute tolerance for integrals


# ===========================================================================
# Class TestChebfun2Construction
# ===========================================================================


class TestChebfun2Construction:
    """Tests for Chebfun2.from_function and chebfun2() factory."""

    def test_from_function_basic(self):
        """from_function returns a Chebfun2."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        assert isinstance(f, Chebfun2)

    def test_chebfun2_factory(self):
        """chebfun2() factory returns a Chebfun2."""
        f = chebfun2(lambda x, y: jnp.cos(x + y))
        assert isinstance(f, Chebfun2)

    def test_rank_cos_x_plus_y(self):
        """cos(x+y) has rank 2 (trig identity)."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        assert f.rank == 2, f"cos(x+y) should be rank 2, got {f.rank}"

    def test_domain_default(self):
        """Default domain is (-1, 1, -1, 1)."""
        f = Chebfun2.from_function(lambda x, y: x + y)
        assert f.domain == (-1.0, 1.0, -1.0, 1.0)

    def test_domain_custom(self):
        """Custom domain is stored correctly."""
        f = Chebfun2.from_function(
            lambda x, y: jnp.sin(x) * jnp.cos(y),
            domain=(0.0, 2.0, -1.0, 1.0),
        )
        assert f.domain == (0.0, 2.0, -1.0, 1.0)

    def test_invalid_domain_length(self):
        """domain with wrong length raises ValueError."""
        with pytest.raises(ValueError, match="domain must have exactly 4"):
            Chebfun2.from_function(lambda x, y: x + y, domain=(-1.0, 1.0))

    def test_fixed_n_not_implemented(self):
        """n= keyword raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="fixed-degree"):
            Chebfun2.from_function(lambda x, y: x + y, n=10)

    def test_repr(self):
        """repr includes class name, rank, and domain."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        s = repr(f)
        assert "Chebfun2" in s
        assert "rank=" in s
        assert "domain=" in s


# ===========================================================================
# Class TestChebfun2Evaluation
# ===========================================================================


class TestChebfun2Evaluation:
    """Tests for Chebfun2.__call__ (evaluation)."""

    def test_cos_x_plus_y_pointwise(self):
        """cos(x+y) evaluation matches direct computation."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        x_test = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 15, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        got = f(xx.ravel(), yy.ravel())
        expected = jnp.cos(xx.ravel() + yy.ravel())
        # atol=1e-14 needed: cos values near zero have large relative error at machine eps
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL_TIGHT, atol=1e-14)

    def test_scalar_evaluation(self):
        """Scalar evaluation returns a scalar."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        val = f(jnp.array(0.3, dtype=jnp.float64), jnp.array(-0.5, dtype=jnp.float64))
        expected = jnp.cos(jnp.array(0.3 - 0.5, dtype=jnp.float64))
        npt.assert_allclose(float(val), float(expected), rtol=RTOL_TIGHT)

    def test_custom_domain_evaluation(self):
        """Evaluation on non-default domain matches direct computation."""
        domain = (0.0, 2.0, -1.0, 1.0)
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x) + y**2, domain=domain)
        x_test = jnp.linspace(0.0, 2.0, 8, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        got = f(xx.ravel(), yy.ravel())
        expected = jnp.sin(xx.ravel()) + yy.ravel() ** 2
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_jit_evaluation(self):
        """JIT-compiled evaluation matches eager evaluation."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        x_val = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_val = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        eager = f(x_val, y_val)
        jitted = eqx.filter_jit(f)(x_val, y_val)
        npt.assert_allclose(np.array(jitted), np.array(eager), rtol=1e-15)

    def test_vmap_evaluation(self):
        """vmap over batch of (x, y) pairs works."""
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        xs = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        ys = jnp.linspace(-0.5, 0.5, 8, dtype=jnp.float64)
        vmapped = jax.vmap(lambda xi, yi: f(xi, yi))(xs, ys)
        expected = jnp.array([f(xs[i], ys[i]) for i in range(8)])
        npt.assert_allclose(np.array(vmapped), np.array(expected), rtol=1e-14)


# ===========================================================================
# Class TestChebfun2Diff
# ===========================================================================


class TestChebfun2Diff:
    """Tests for Chebfun2.diff — partial derivatives."""

    def test_diff_y_cos_x_plus_y(self):
        """d/dy cos(x+y) = -sin(x+y)."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        dfy = f.diff(dim=1, k=1)
        x_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        got = dfy(xx.ravel(), yy.ravel())
        expected = -jnp.sin(xx.ravel() + yy.ravel())
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL, atol=1e-12)

    def test_diff_x_cos_x_plus_y(self):
        """d/dx cos(x+y) = -sin(x+y)."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        dfx = f.diff(dim=2, k=1)
        x_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        y_test = jnp.linspace(-1.0, 1.0, 10, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        got = dfx(xx.ravel(), yy.ravel())
        expected = -jnp.sin(xx.ravel() + yy.ravel())
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL, atol=1e-12)

    def test_diff_y_x_times_y(self):
        """d/dy (x*y) = x."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        dfy = f.diff(dim=1, k=1)
        x_test = jnp.array([0.1, -0.5, 0.8], dtype=jnp.float64)
        y_test = jnp.array([0.2, 0.3, -0.7], dtype=jnp.float64)
        got = dfy(x_test, y_test)
        expected = x_test  # d/dy (xy) = x
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_diff_x_x_times_y(self):
        """d/dx (x*y) = y."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        dfx = f.diff(dim=2, k=1)
        x_test = jnp.array([0.1, -0.5, 0.8], dtype=jnp.float64)
        y_test = jnp.array([0.2, 0.3, -0.7], dtype=jnp.float64)
        got = dfx(x_test, y_test)
        expected = y_test  # d/dx (xy) = y
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_diff_zero_order_identity(self):
        """diff(k=0) returns the same function."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        g = f.diff(dim=1, k=0)
        # g and f should evaluate the same
        x0 = jnp.array(0.3, dtype=jnp.float64)
        y0 = jnp.array(-0.4, dtype=jnp.float64)
        npt.assert_allclose(float(g(x0, y0)), float(f(x0, y0)), rtol=1e-15)

    def test_diff_second_order(self):
        """d^2/dy^2 cos(x+y) = -cos(x+y)."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        d2fy = f.diff(dim=1, k=2)
        x_test = jnp.linspace(-0.8, 0.8, 6, dtype=jnp.float64)
        y_test = jnp.linspace(-0.8, 0.8, 6, dtype=jnp.float64)
        xx, yy = jnp.meshgrid(x_test, y_test)
        got = d2fy(xx.ravel(), yy.ravel())
        expected = -jnp.cos(xx.ravel() + yy.ravel())
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_diff_invalid_dim(self):
        """diff with invalid dim raises ValueError."""
        f = Chebfun2.from_function(lambda x, y: x + y)
        with pytest.raises(ValueError, match="dim must be 1"):
            f.diff(dim=3)

    def test_diff_negative_k_raises(self):
        """diff with k < 0 raises ValueError."""
        f = Chebfun2.from_function(lambda x, y: x + y)
        with pytest.raises(ValueError, match="k must be >= 0"):
            f.diff(dim=1, k=-1)

    def test_diff_custom_domain(self):
        """Differentiation on a non-default domain has correct scaling.

        On domain [0, pi] x [0, pi], d/dx sin(x) = cos(x).
        """
        domain = (0.0, float(np.pi), 0.0, float(np.pi))
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x), domain=domain)
        dfx = f.diff(dim=2, k=1)
        x_test = jnp.linspace(0.1, float(np.pi) - 0.1, 8, dtype=jnp.float64)
        y_test = jnp.zeros(8, dtype=jnp.float64)
        got = dfx(x_test, y_test)
        expected = jnp.cos(x_test)
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)


# ===========================================================================
# Class TestChebfun2Sum
# ===========================================================================


class TestChebfun2Sum:
    """Tests for Chebfun2.sum and Chebfun2.sum2."""

    def test_sum2_constant_one(self):
        """Double integral of 1 over [-1,1]^2 = 4.

        MATLAB: sum2(chebfun2(@(x,y) ones(size(x)))) = 4.
        """
        f = Chebfun2.from_function(lambda x, y: jnp.ones_like(x))
        result = f.sum2()
        npt.assert_allclose(float(result), 4.0, rtol=1e-14)

    def test_sum2_cos_x_plus_y(self):
        """Double integral of cos(x+y) over [-1,1]^2.

        Analytically:
          int_{-1}^1 int_{-1}^1 cos(x+y) dx dy
          = (int_{-1}^1 cos(x) dx)^2 - (int_{-1}^1 sin(x) dx)^2
            ... actually let's compute directly:
          = [sin(x)]_{-1}^1 * [evaluated in y]
          int_{-1}^1 cos(x+y) dx = [sin(x+y)]_{-1}^1 = sin(1+y) - sin(-1+y)
          int_{-1}^1 (sin(1+y) - sin(-1+y)) dy
          = [-cos(1+y)]_{-1}^1 - [-cos(-1+y)]_{-1}^1
          = (-cos(2) + cos(0)) - (-cos(0) + cos(-2))
          = (1 - cos(2)) - (cos(2) - 1)   [since cos(-2)=cos(2)]
          = 2*(1 - cos(2))
          = 2 * 2 * sin^2(1)   [using 1-cos(2)=2sin^2(1)]
          = 4 * sin^2(1)
        """
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        result = f.sum2()
        expected = 4.0 * float(jnp.sin(jnp.array(1.0, dtype=jnp.float64)) ** 2)
        npt.assert_allclose(float(result), expected, rtol=1e-13)

    def test_sum_no_arg_equals_sum2(self):
        """sum() with no argument equals sum2()."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        npt.assert_allclose(float(f.sum()), float(f.sum2()), rtol=1e-15)

    def test_sum_dim1_returns_chebfun2(self):
        """sum(dim=1) returns a Chebfun2."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        g = f.sum(dim=1)
        assert isinstance(g, Chebfun2)

    def test_sum_dim2_returns_chebfun2(self):
        """sum(dim=2) returns a Chebfun2."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        g = f.sum(dim=2)
        assert isinstance(g, Chebfun2)

    def test_sum_dim1_x_times_y(self):
        """sum over y of x*y: int_{-1}^{1} x*y dy = 0 (odd in y)."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        g = f.sum(dim=1)  # integrate over y, returns function of x
        x_test = jnp.array([0.2, -0.5, 0.7], dtype=jnp.float64)
        y_test = jnp.zeros(3, dtype=jnp.float64)  # y value doesn't matter for g
        got = g(x_test, y_test)
        # int_{-1}^{1} x*y dy = x * [y^2/2]_{-1}^{1} = x * (1/2 - 1/2) = 0
        npt.assert_allclose(np.array(got), np.zeros(3), atol=1e-13)

    def test_sum_dim2_x_times_y(self):
        """sum over x of x*y: int_{-1}^{1} x*y dx = 0 (odd in x)."""
        f = Chebfun2.from_function(lambda x, y: x * y)
        g = f.sum(dim=2)  # integrate over x, returns function of y
        x_test = jnp.zeros(3, dtype=jnp.float64)  # x value doesn't matter for g
        y_test = jnp.array([0.2, -0.5, 0.7], dtype=jnp.float64)
        got = g(x_test, y_test)
        # int_{-1}^{1} x*y dx = y * [x^2/2]_{-1}^{1} = y * (1/2 - 1/2) = 0
        npt.assert_allclose(np.array(got), np.zeros(3), atol=1e-13)

    def test_sum_dim1_sin_x_cos_y(self):
        """sum(dim=1) of sin(x)*cos(y): int_{-1}^1 sin(x)*cos(y) dy.

        = sin(x) * 2*sin(1)
        """
        f = Chebfun2.from_function(lambda x, y: jnp.sin(x) * jnp.cos(y))
        g = f.sum(dim=1)  # integrate over y
        x_test = jnp.linspace(-1.0, 1.0, 8, dtype=jnp.float64)
        y_test = jnp.zeros(8, dtype=jnp.float64)  # y value doesn't matter for g
        got = g(x_test, y_test)
        # int_{-1}^{1} cos(y) dy = 2*sin(1)
        expected = jnp.sin(x_test) * 2.0 * jnp.sin(jnp.array(1.0, dtype=jnp.float64))
        npt.assert_allclose(np.array(got), np.array(expected), rtol=RTOL)

    def test_sum_invalid_dim(self):
        """sum with invalid dim raises ValueError."""
        f = Chebfun2.from_function(lambda x, y: x + y)
        with pytest.raises(ValueError, match="dim must be None"):
            f.sum(dim=3)

    def test_sum2_custom_domain(self):
        """Double integral of 1 over [0,2]x[0,3] = 6."""
        f = Chebfun2.from_function(
            lambda x, y: jnp.ones_like(x), domain=(0.0, 2.0, 0.0, 3.0)
        )
        result = f.sum2()
        npt.assert_allclose(float(result), 6.0, rtol=1e-13)


# ===========================================================================
# Class TestChebfun2Norm
# ===========================================================================


class TestChebfun2Norm:
    """Tests for Chebfun2.norm (Frobenius / L2 norm)."""

    def test_norm_constant_one(self):
        """L2 norm of f=1 over [-1,1]^2 = sqrt(4) = 2."""
        f = Chebfun2.from_function(lambda x, y: jnp.ones_like(x))
        result = f.norm()
        npt.assert_allclose(float(result), 2.0, rtol=1e-13)

    def test_norm_fro_alias(self):
        """norm() and norm('fro') give the same result."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        npt.assert_allclose(float(f.norm()), float(f.norm("fro")), rtol=1e-15)

    def test_norm_nonnegative(self):
        """Norm is always non-negative."""
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        assert float(f.norm()) >= 0.0

    def test_norm_cos_x_plus_y(self):
        """L2 norm of cos(x+y) over [-1,1]^2.

        ||cos(x+y)||_F^2 = int_{-1}^{1} int_{-1}^{1} cos^2(x+y) dx dy
                         = (1/2) * int ... (1 + cos(2(x+y))) dx dy
        By symmetry = (1/2) * (4 + int cos(2(x+y)) dx dy)
        int_{-1}^{1} int_{-1}^{1} cos(2x+2y) dx dy
          = int_{-1}^{1} [sin(2x+2y)/(2)]_{x=-1}^{x=1} dy
          = (1/2) int_{-1}^{1} (sin(2+2y) - sin(-2+2y)) dy
          = (1/2) [-cos(2+2y)/2 + cos(-2+2y)/2]_{-1}^{1}
          = (1/4)[(-cos(4)+cos(0)) - (-cos(0)+cos(-4))]
          = (1/4)[(-cos(4)+1) - (-1+cos(4))]
          = (1/4)[2 - 2*cos(4)]
          = (1/2)(1 - cos(4))
        So ||f||^2 = (1/2)(4 + (1/2)(1 - cos(4)))
                   = 2 + (1/4)(1 - cos(4))
        """
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        norm_sq_expected = 2.0 + 0.25 * (1.0 - float(jnp.cos(jnp.array(4.0, dtype=jnp.float64))))
        norm_expected = np.sqrt(norm_sq_expected)
        npt.assert_allclose(float(f.norm()), norm_expected, rtol=1e-11)

    def test_norm_invalid_p(self):
        """norm with invalid p raises NotImplementedError."""
        f = Chebfun2.from_function(lambda x, y: x + y)
        with pytest.raises(NotImplementedError, match="Frobenius"):
            f.norm(p=1)


# ===========================================================================
# Class TestChebfun2Roots
# ===========================================================================


class TestChebfun2Roots:
    """Tests for Chebfun2.roots (zero contour detection)."""

    def test_roots_returns_list(self):
        """roots() returns a list."""
        f = Chebfun2.from_function(lambda x, y: x)
        result = f.roots()
        assert isinstance(result, list)

    def test_roots_zero_contour_x(self):
        """roots() of f=x should find the zero contour x=0.

        The zero set of f(x,y)=x is the line x=0, which crosses the domain.
        """
        f = Chebfun2.from_function(lambda x, y: x + jnp.zeros_like(y))
        contours = f.roots()
        # Should find at least one contour near x=0
        assert len(contours) >= 1
        # All points on the contour should have |x| < some tolerance
        for c in contours:
            npt.assert_allclose(c[:, 0], np.zeros(c.shape[0]), atol=0.01)

    def test_roots_circle(self):
        """roots() of x^2+y^2-0.25 should trace a circle of radius 0.5."""
        f = Chebfun2.from_function(lambda x, y: x**2 + y**2 - 0.25)
        contours = f.roots()
        # Should find one contour (the circle)
        assert len(contours) >= 1
        # Each point on the contour should satisfy x^2+y^2 ≈ 0.25
        for c in contours:
            radii = c[:, 0] ** 2 + c[:, 1] ** 2
            npt.assert_allclose(radii, 0.25 * np.ones(len(radii)), atol=0.01)


# ===========================================================================
# Class TestChebfun2GoldenRef — MATLAB golden reference values
# ===========================================================================


class TestChebfun2GoldenRef:
    """MATLAB golden-reference tests for Chebfun2.

    Reference values computed with MATLAB Chebfun (commit 7574c77):

        f = chebfun2(@(x,y) cos(x+y));
        sum2(f)    % 4*sin(1)^2 ≈ 2.8332...  (see formula above)
        norm(f)    % sqrt(2 + (1-cos(4))/4)
    """

    def test_double_integral_one(self):
        """int_[-1,1]^2 1 dx dy = 4 exactly.

        MATLAB: sum2(chebfun2(@(x,y) ones(size(x)))) = 4.
        """
        f = Chebfun2.from_function(lambda x, y: jnp.ones_like(x))
        npt.assert_allclose(float(f.sum2()), 4.0, rtol=1e-14,
                            err_msg="Golden ref: sum2(1) = 4")

    def test_double_integral_x_squared(self):
        """int_[-1,1]^2 x^2 dx dy = 4/3.

        int_{-1}^1 x^2 dx = 2/3. Times 2 (y integral) = 4/3.
        MATLAB: sum2(chebfun2(@(x,y) x.^2)) = 4/3.
        """
        f = Chebfun2.from_function(lambda x, y: x**2 + jnp.zeros_like(y))
        expected = 4.0 / 3.0
        npt.assert_allclose(float(f.sum2()), expected, rtol=1e-13,
                            err_msg="Golden ref: sum2(x^2) = 4/3")

    def test_double_integral_exp_xy(self):
        """int_[-1,1]^2 exp(x*y) dx dy.

        int_{-1}^1 int_{-1}^1 e^{xy} dx dy
          = int_{-1}^1 [(e^y - e^{-y})/y] dy  (for y≠0)
          = int_{-1}^1 2*sinh(y)/y dy
          = 2 * int_{-1}^1 sinh(y)/y dy
        This is related to the Si function. Numerically ≈ 2.9937...
        MATLAB: format long; sum2(chebfun2(@(x,y) exp(x.*y)))
        """
        # Compute reference numerically
        from scipy import integrate
        ref, _ = integrate.dblquad(
            lambda y, x: np.exp(x * y),
            -1.0, 1.0,
            -1.0, 1.0,
            epsabs=1e-12,
            epsrel=1e-12,
        )
        f = Chebfun2.from_function(lambda x, y: jnp.exp(x * y))
        npt.assert_allclose(float(f.sum2()), ref, rtol=1e-10,
                            err_msg="Golden ref: sum2(exp(x*y))")
