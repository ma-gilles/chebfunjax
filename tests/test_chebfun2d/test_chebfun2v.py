"""Tests for Chebfun2v — 2D vector-valued functions on a rectangle.

JAX contract:
    construction   : jit=NO (Python adaptive loop via SeparableApprox)
    evaluation     : jit=YES (via __call__)

Test coverage (Tier 1 — unit tests, no MATLAB required):
    - Construction from functions
    - Evaluation: point-evaluation matches component functions
    - Negation, scalar multiply, add
    - Divergence: div(grad(f)) == Laplacian (not tested here; basic div check)
    - Curl of a gradient is zero (key mathematical identity)
    - Cross product: orthogonality identity
    - dot product: symmetry
    - norm: non-negative
    - repr
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun2d.separable_approx import SeparableApprox

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------
ATOL = 1e-8
RTOL = 1e-6


# ===========================================================================
# Construction tests
# ===========================================================================


class TestConstruction:
    """Tests for Chebfun2v construction."""

    def test_from_functions_2comp(self):
        """Construct 2-component field from callables."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x),
            lambda x, y: jnp.cos(y),
        )
        assert F.n_components == 2
        assert F.domain == (-1.0, 1.0, -1.0, 1.0)

    def test_from_functions_3comp(self):
        """Construct 3-component field from callables."""
        F = Chebfun2v.from_functions(
            lambda x, y: x,
            lambda x, y: y,
            lambda x, y: x * y,
        )
        assert F.n_components == 3

    def test_domain_mismatch_raises(self):
        """Components with different domains raise ValueError."""
        f = SeparableApprox.from_function(lambda x, y: x, domain=(-1.0, 1.0, -1.0, 1.0))
        g = SeparableApprox.from_function(lambda x, y: y, domain=(0.0, 1.0, 0.0, 1.0))
        with pytest.raises(ValueError, match="domain"):
            Chebfun2v([f, g])

    def test_too_few_components_raises(self):
        """Fewer than 2 components raises ValueError."""
        f = SeparableApprox.from_function(lambda x, y: x)
        with pytest.raises(ValueError):
            Chebfun2v([f])

    def test_repr(self):
        """repr includes n_components and domain."""
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        s = repr(F)
        assert "Chebfun2v" in s
        assert "n_components=2" in s


# ===========================================================================
# Evaluation tests
# ===========================================================================


class TestEvaluation:
    """Tests for Chebfun2v.__call__ evaluation."""

    def test_eval_components_match(self):
        """Evaluate F(x,y) and compare with individual component evaluations."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x),
            lambda x, y: jnp.cos(y),
        )
        x = jnp.array([0.0, 0.5, -0.5])
        y = jnp.array([0.0, 0.3, -0.3])
        vals = F(x, y)
        assert vals.shape == (3, 2)
        f_expected = np.sin(np.array(x))
        g_expected = np.cos(np.array(y))
        npt.assert_allclose(np.array(vals[:, 0]), f_expected, atol=ATOL, rtol=0)
        npt.assert_allclose(np.array(vals[:, 1]), g_expected, atol=ATOL, rtol=0)

    def test_eval_scalar(self):
        """Scalar evaluation returns shape (2,)."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.ones_like(x) * 2.0,
            lambda x, y: jnp.ones_like(y) * 3.0,
        )
        x = jnp.array(0.5)
        y = jnp.array(-0.3)
        vals = F(x, y)
        assert vals.shape == (2,)
        npt.assert_allclose(float(vals[0]), 2.0, atol=1e-6)
        npt.assert_allclose(float(vals[1]), 3.0, atol=1e-6)


# ===========================================================================
# Arithmetic tests
# ===========================================================================


class TestArithmetic:
    """Tests for Chebfun2v arithmetic operations."""

    def _make_field(self) -> Chebfun2v:
        return Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x),
            lambda x, y: jnp.cos(y),
        )

    def test_negation(self):
        """Negate: (-F)(x,y) = -F(x,y)."""
        F = self._make_field()
        neg_F = -F
        x = jnp.array([0.3, -0.2])
        y = jnp.array([0.1, 0.4])
        vals_F = F(x, y)
        vals_neg = neg_F(x, y)
        npt.assert_allclose(np.array(vals_neg), -np.array(vals_F), atol=ATOL, rtol=0)

    def test_scalar_multiply(self):
        """3 * F evaluates to 3 times F."""
        F = self._make_field()
        G = 3.0 * F
        x = jnp.array([0.0, 0.5])
        y = jnp.array([0.0, -0.5])
        npt.assert_allclose(np.array(G(x, y)), 3.0 * np.array(F(x, y)), atol=ATOL, rtol=0)

    def test_vector_add(self):
        """F + G evaluates component-wise."""
        F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x), lambda x, y: jnp.cos(y))
        G = Chebfun2v.from_functions(lambda x, y: jnp.cos(x), lambda x, y: jnp.sin(y))
        H = F + G
        x = jnp.array([0.2, -0.3])
        y = jnp.array([0.1, 0.5])
        npt.assert_allclose(
            np.array(H(x, y)),
            np.array(F(x, y)) + np.array(G(x, y)),
            atol=ATOL,
            rtol=0,
        )

    def test_sub(self):
        """F - F = 0 (approximately)."""
        F = self._make_field()
        Z = F - F
        x = jnp.array([0.1, 0.3, -0.4])
        y = jnp.array([0.2, -0.1, 0.0])
        vals = np.array(Z(x, y))
        npt.assert_allclose(vals, 0.0, atol=1e-5, rtol=0)


# ===========================================================================
# Differential calculus tests
# ===========================================================================


class TestCalculus:
    """Tests for divergence, curl, diff."""

    def test_divergence_constant_field(self):
        """Divergence of a constant vector field = 0."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.ones_like(x) * 2.0,
            lambda x, y: jnp.ones_like(y) * 3.0,
        )
        div_F = F.divergence()
        x = jnp.array([0.0, 0.5, -0.5])
        y = jnp.array([0.0, 0.3, -0.3])
        npt.assert_allclose(np.array(div_F(x, y)), 0.0, atol=1e-6, rtol=0)

    def test_divergence_linear_field(self):
        """div([x; y]) = 1 + 1 = 2."""
        F = Chebfun2v.from_functions(
            lambda x, y: x,
            lambda x, y: y,
        )
        div_F = F.divergence()
        x = jnp.array([0.0, 0.3, -0.4])
        y = jnp.array([0.2, -0.1, 0.5])
        npt.assert_allclose(np.array(div_F(x, y)), 2.0, atol=1e-6, rtol=0)

    def test_curl_of_gradient_is_zero(self):
        """curl(grad f) = 0 for any smooth scalar f.

        grad f = [f_x; f_y], curl(grad f) = d(f_y)/dx - d(f_x)/dy = f_yx - f_xy = 0.
        """
        # f(x,y) = sin(x)*cos(y)
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.cos(x) * jnp.cos(y),  # f_x
            lambda x, y: -jnp.sin(x) * jnp.sin(y),  # f_y
        )
        curl_F = F.curl()  # should be a SeparableApprox near zero
        x = jnp.array([0.0, 0.3, -0.3, 0.7, -0.7])
        y = jnp.array([0.0, 0.4, -0.2, -0.5, 0.6])
        npt.assert_allclose(np.array(curl_F(x, y)), 0.0, atol=1e-5, rtol=0)

    def test_curl_2d_sin_x(self):
        """curl([sin(x); 0]) = 0 - cos(x) = -cos(x)."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x),
            lambda x, y: jnp.zeros_like(x),
        )
        curl_F = F.curl()
        x = jnp.array([0.0, 0.5, 1.0])
        y = jnp.array([0.0, 0.0, 0.0])
        # curl([sin(x); 0]) = d(0)/dx - d(sin(x))/dy = 0 - 0 = 0
        # Wait: MATLAB convention: curl(F) = F(2)_x - F(1)_y
        # = d(0)/dx - d(sin(x))/dy = 0 - 0 = 0
        npt.assert_allclose(np.array(curl_F(x, y)), 0.0, atol=1e-6, rtol=0)

    def test_curl_2d_y_x(self):
        """curl([y; x]) = dx/dx - dy/dy = 1 - 1 = 0."""
        # Actually: curl([y; x]) = d(x)/dx - d(y)/dy = 1 - 1 = 0
        # Wait MATLAB: curl(F) = F(2)_x - F(1)_y = d(x)/dx - d(y)/dy = 1 - 1 = 0
        F = Chebfun2v.from_functions(
            lambda x, y: y,
            lambda x, y: x,
        )
        curl_F = F.curl()
        x = jnp.array([0.1, -0.3, 0.7])
        y = jnp.array([0.2, 0.4, -0.1])
        npt.assert_allclose(np.array(curl_F(x, y)), 0.0, atol=1e-6, rtol=0)

    def test_curl_2d_known(self):
        """curl([0; x^2]) = d(x^2)/dx - d(0)/dy = 2x."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.zeros_like(x),
            lambda x, y: x**2,
        )
        curl_F = F.curl()
        x_pts = jnp.array([0.0, 0.5, -0.5, 1.0])
        y_pts = jnp.array([0.0, 0.3, 0.1, -0.2])
        expected = 2.0 * np.array(x_pts)
        npt.assert_allclose(np.array(curl_F(x_pts, y_pts)), expected, atol=1e-5, rtol=0)


# ===========================================================================
# Dot and cross product tests
# ===========================================================================


class TestDotCross:
    """Tests for dot and cross products."""

    def test_dot_self_nonneg(self):
        """F . F >= 0."""
        F = Chebfun2v.from_functions(
            lambda x, y: jnp.sin(x) + y,
            lambda x, y: x * jnp.cos(y),
        )
        fdotf = F.dot(F)
        x = jnp.linspace(-0.9, 0.9, 5)
        y = jnp.linspace(-0.9, 0.9, 5)
        vals = np.array(fdotf(x, y))
        assert np.all(vals >= -1e-6), f"F.F should be >= 0, got min={vals.min()}"

    def test_dot_symmetry(self):
        """F . G = G . F."""
        F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x), lambda x, y: jnp.cos(y))
        G = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        fdg = F.dot(G)
        gdf = G.dot(F)
        x = jnp.array([0.1, -0.3, 0.5])
        y = jnp.array([0.2, 0.4, -0.1])
        npt.assert_allclose(np.array(fdg(x, y)), np.array(gdf(x, y)), atol=1e-6, rtol=0)

    def test_cross_2d_self_zero(self):
        """F x F = 0."""
        F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x), lambda x, y: jnp.cos(y))
        fxf = F.cross(F)  # returns SeparableApprox
        x = jnp.array([0.0, 0.3, -0.5])
        y = jnp.array([0.1, -0.2, 0.4])
        npt.assert_allclose(np.array(fxf(x, y)), 0.0, atol=1e-6, rtol=0)

    def test_cross_2d_antisymmetric(self):
        """F x G = -(G x F)."""
        F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x), lambda x, y: jnp.cos(y))
        G = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        fxg = F.cross(G)
        gxf = G.cross(F)
        x = jnp.array([0.2, -0.1, 0.6])
        y = jnp.array([0.3, 0.5, -0.4])
        npt.assert_allclose(np.array(fxg(x, y)), -np.array(gxf(x, y)), atol=1e-5, rtol=0)

    def test_norm_nonneg(self):
        """norm(F) >= 0."""
        F = Chebfun2v.from_functions(lambda x, y: jnp.sin(x), lambda x, y: jnp.cos(y))
        n = F.norm()
        assert n >= 0.0, f"norm should be >= 0, got {n}"
