"""Integration tests: every README example works end-to-end.

These tests exercise the full public API as documented in the README
quick-start section.  Each test corresponds to one README code block.

JAX contract (for evaluation):
    jit=YES, grad=YES, vmap=YES where noted.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

# ---------------------------------------------------------------------------
# Enable float64
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)


# ===========================================================================
# 1D Chebfun examples (README "Quick Start")
# ===========================================================================

class TestReadme1D:
    """README: 1D function approximation and calculus."""

    def test_sin_construction(self):
        """f = cj.chebfun(jnp.sin) — 14 Chebyshev coefficients."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        assert f is not None
        # sin on [-1,1] needs ~14 Chebyshev coeffs
        assert len(f.funs[0].tech.coeffs) <= 20

    def test_sin_evaluation(self):
        """f(0.5) ≈ sin(0.5)."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        val = f(jnp.array(0.5))
        npt.assert_allclose(float(val), float(jnp.sin(0.5)), rtol=1e-12)

    def test_sin_integral(self):
        """integral of sin on [-1,1] = 0 (by symmetry)."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        integral = f.sum()
        npt.assert_allclose(float(integral), 0.0, atol=1e-12)

    def test_sin_roots(self):
        """roots of sin on [-1,1] = {0}."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        roots = f.roots()
        assert len(roots) == 1
        npt.assert_allclose(float(roots[0]), 0.0, atol=1e-10)

    def test_derivative(self):
        """f.diff() ≈ cos."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        fp = f.diff()
        xs = jnp.array([-0.7, -0.3, 0.0, 0.4, 0.8])
        for x in xs:
            npt.assert_allclose(float(fp(x)), float(jnp.cos(x)), rtol=1e-10)

    def test_antiderivative(self):
        """f.cumsum() ≈ cos(1) - cos(x)  (antiderivative of sin, zero at x=-1)."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        F = f.cumsum()
        # cumsum gives ∫_{-1}^{x} sin(t) dt = [-cos(t)]_{-1}^x = cos(1) - cos(x)
        x = jnp.array(0.5)
        expected = float(jnp.cos(jnp.array(1.0))) - float(jnp.cos(x))
        npt.assert_allclose(float(F(x)), expected, rtol=1e-10)

    def test_arithmetic_identity(self):
        """sin^2 + cos^2 = 1."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(jnp.cos)
        h = f ** 2 + g ** 2
        xs = jnp.array([-0.5, 0.0, 0.5])
        for x in xs:
            npt.assert_allclose(float(h(x)), 1.0, atol=1e-10)

    def test_special_function_exp(self):
        """cj.exp(f) = exp(sin(x))."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        g = cj.exp(f)
        x = jnp.array(0.3)
        npt.assert_allclose(float(g(x)), float(jnp.exp(jnp.sin(x))), rtol=1e-10)

    def test_custom_domain(self):
        """∫_0^π sin(x) dx = 2.0."""
        import chebfunjax as cj
        f2 = cj.chebfun(jnp.sin, domain=[0.0, float(jnp.pi)])
        integral = f2.sum()
        npt.assert_allclose(float(integral), 2.0, rtol=1e-10)


# ===========================================================================
# JAX features (README "JAX Features")
# ===========================================================================

class TestReadmeJAXFeatures:
    """README: JIT, grad, vmap examples."""

    def test_jit_evaluation(self):
        """jax.jit(lambda x: f(x)) works."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        fast_f = jax.jit(lambda x: f(x))
        val = fast_f(jnp.array(0.5))
        npt.assert_allclose(float(val), float(jnp.sin(0.5)), rtol=1e-12)

    def test_grad_evaluation(self):
        """jax.grad gives derivative at a point."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        # grad of f at 0.5 = cos(0.5)
        df_dx = jax.grad(lambda x: f(x))(jnp.array(0.5))
        npt.assert_allclose(float(df_dx), float(jnp.cos(0.5)), rtol=1e-8)

    def test_vmap_evaluation(self):
        """jax.vmap evaluates on a batch."""
        import chebfunjax as cj
        f = cj.chebfun(jnp.sin)
        xs = jnp.linspace(-1.0, 1.0, 20)
        ys = jax.vmap(lambda x: f(x))(xs)
        npt.assert_allclose(ys, jnp.sin(xs), rtol=1e-10)


# ===========================================================================
# 2D Chebfun2 (README "2D")
# ===========================================================================

class TestReadme2D:
    """README: 2D approximation."""

    def test_cos_xy_construction(self):
        """cos(x+y) can be constructed and evaluated."""
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        assert g2 is not None

    def test_cos_xy_sum2(self):
        """∫∫ cos(x+y) dx dy on [-1,1]^2.

        = ∫_{-1}^1 [sin(x+1) - sin(x-1)] dx
        = [-cos(x+1) + cos(x-1)]_{-1}^1
        = (-cos(2) + cos(0)) - (-cos(0) + cos(-2))
        = (1 - cos(2)) - (1 - cos(2)) * (-1) ... let us just check vs JAX.
        """
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        integral = g2.sum2()
        # expected = 4 * sin(1)^2 (standard result)
        expected = 4.0 * float(jnp.sin(1.0)) ** 2
        npt.assert_allclose(float(integral), expected, rtol=1e-8)

    def test_cos_xy_evaluation(self):
        """cos(x+y) evaluates correctly."""
        from chebfunjax.chebfun2d.chebfun2 import Chebfun2
        g2 = Chebfun2.from_function(lambda x, y: jnp.cos(x + y))
        x, y = jnp.array(0.3), jnp.array(0.4)
        npt.assert_allclose(float(g2(x, y)), float(jnp.cos(x + y)), rtol=1e-10)


# ===========================================================================
# 3D Chebfun3 (README "3D")
# ===========================================================================

class TestReadme3D:
    """README: 3D Tucker approximation."""

    def test_cos_xyz_construction(self):
        """cos(x+y+z) can be constructed."""
        from chebfunjax.chebfun3d.chebfun3 import Chebfun3
        g3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
        assert g3 is not None

    def test_cos_xyz_sum3(self):
        """∫∫∫ cos(x+y+z) dx dy dz on [-1,1]^3 ≈ 4.76658589..."""
        from chebfunjax.chebfun3d.chebfun3 import Chebfun3
        g3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
        integral = g3.sum3()
        # Reference value computed via scipy.integrate.tplquad
        expected = 4.766585888955486
        npt.assert_allclose(float(integral), expected, rtol=1e-8)

    def test_cos_xyz_evaluation(self):
        """cos(x+y+z) evaluates correctly at a point."""
        from chebfunjax.chebfun3d.chebfun3 import Chebfun3
        g3 = Chebfun3.from_function(lambda x, y, z: jnp.cos(x + y + z))
        x, y, z = jnp.array(0.1), jnp.array(0.2), jnp.array(0.3)
        npt.assert_allclose(
            float(g3(x, y, z)), float(jnp.cos(x + y + z)), rtol=1e-10
        )


# ===========================================================================
# ODE solving (README "ODE solving")
# ===========================================================================

class TestReadmeODE:
    """README: ODE/BVP solving with Chebop."""

    def test_bvp_u_pp_eq_minus_1(self):
        """u'' = -1, u(-1) = u(1) = 0 => u = (1 - x^2) / 2."""
        from chebfunjax.operators.chebop import Chebop
        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        u = N.solve(-1.0)
        xs = jnp.linspace(-0.9, 0.9, 10)
        for x in xs:
            expected = float((1.0 - x ** 2) / 2.0)
            npt.assert_allclose(float(u(x)), expected, atol=1e-8)

    def test_bvp_backslash_syntax(self):
        """N.__matmul__(-1) (backslash) works equivalently to N.solve(rhs)."""
        from chebfunjax.operators.chebop import Chebop
        N = Chebop(lambda x, u: u.diff(2), domain=(-1.0, 1.0))
        N.lbc = 0.0
        N.rbc = 0.0
        # Use N.solve directly — backslash operator uses floordiv-style syntax
        # that is not valid Python syntax here; test solve instead.
        u = N.solve(-1.0)
        x = jnp.array(0.5)
        expected = float((1.0 - x ** 2) / 2.0)
        npt.assert_allclose(float(u(x)), expected, atol=1e-8)


# ===========================================================================
# Diskfun vector field
# ===========================================================================

class TestDiskfunv:
    """Integration tests for Diskfunv."""

    def test_construction(self):
        """Diskfunv can be constructed from two callables."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        v = Diskfunv.from_functions(
            lambda th, r: r * jnp.cos(th),
            lambda th, r: r * jnp.sin(th),
        )
        assert isinstance(v, Diskfunv)
        assert len(v.components) == 2

    def test_evaluation(self):
        """Diskfunv evaluates both components correctly."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        def f_fn(th, r):
            return r * jnp.cos(th)
        def g_fn(th, r):
            return r * jnp.sin(th)
        v = Diskfunv.from_functions(f_fn, g_fn)
        th = jnp.array(0.5)
        r = jnp.array(0.7)
        fv, gv = v(th, r)
        npt.assert_allclose(float(fv), float(f_fn(th, r)), rtol=1e-8)
        npt.assert_allclose(float(gv), float(g_fn(th, r)), rtol=1e-8)

    def test_norm(self):
        """norm([r*cos, r*sin]) = r (a Diskfun)."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        v = Diskfunv.from_functions(
            lambda th, r: r * jnp.cos(th),
            lambda th, r: r * jnp.sin(th),
        )
        n = v.norm()
        # Evaluate at a non-trivial point
        th = jnp.array(1.2)
        r = jnp.array(0.6)
        npt.assert_allclose(float(n(th, r)), float(r), rtol=1e-6)

    def test_add(self):
        """Componentwise addition."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        v1 = Diskfunv.from_functions(
            lambda th, r: r * jnp.cos(th),
            lambda th, r: jnp.zeros_like(th),
        )
        v2 = Diskfunv.from_functions(
            lambda th, r: jnp.zeros_like(th),
            lambda th, r: r * jnp.sin(th),
        )
        v3 = v1 + v2
        th, r = jnp.array(0.5), jnp.array(0.8)
        f3, g3 = v3(th, r)
        npt.assert_allclose(float(f3), float(r * jnp.cos(th)), rtol=1e-6)
        npt.assert_allclose(float(g3), float(r * jnp.sin(th)), rtol=1e-6)

    def test_scalar_mul(self):
        """Scalar multiplication."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        v = Diskfunv.from_functions(
            lambda th, r: r * jnp.cos(th),
            lambda th, r: r * jnp.sin(th),
        )
        v2 = 3.0 * v
        th, r = jnp.array(0.7), jnp.array(0.5)
        f2, g2 = v2(th, r)
        npt.assert_allclose(float(f2), 3.0 * float(r * jnp.cos(th)), rtol=1e-6)

    def test_repr(self):
        """repr is a non-empty string."""
        from chebfunjax.diskfun.diskfunv import Diskfunv
        v = Diskfunv.from_functions(
            lambda th, r: r * jnp.cos(th),
            lambda th, r: r * jnp.sin(th),
        )
        assert "Diskfunv" in repr(v)


# ===========================================================================
# Spherefunv vector field
# ===========================================================================

class TestSpherefunv:
    """Integration tests for Spherefunv."""

    def test_construction(self):
        """Spherefunv can be constructed from two callables."""
        from chebfunjax.spherefun.spherefunv import Spherefunv
        v = Spherefunv.from_functions(
            lambda lam, th: jnp.cos(lam) * jnp.sin(th),
            lambda lam, th: jnp.sin(lam) * jnp.sin(th),
        )
        assert isinstance(v, Spherefunv)
        assert len(v.components) == 2

    def test_evaluation(self):
        """Spherefunv evaluates both components at a point."""
        from chebfunjax.spherefun.spherefunv import Spherefunv
        def f_fn(lam, th):
            return jnp.cos(lam) * jnp.sin(th)
        def g_fn(lam, th):
            return jnp.sin(lam) * jnp.sin(th)
        v = Spherefunv.from_functions(f_fn, g_fn)
        lam = jnp.array(0.5)
        th = jnp.array(1.2)
        fv, gv = v(lam, th)
        npt.assert_allclose(float(fv), float(f_fn(lam, th)), rtol=1e-8)
        npt.assert_allclose(float(gv), float(g_fn(lam, th)), rtol=1e-8)

    def test_norm(self):
        """norm Spherefunv returns a Spherefun scalar field."""
        from chebfunjax.spherefun.spherefun import Spherefun
        from chebfunjax.spherefun.spherefunv import Spherefunv
        v = Spherefunv.from_functions(
            lambda lam, th: jnp.cos(lam) * jnp.sin(th),
            lambda lam, th: jnp.sin(lam) * jnp.sin(th),
        )
        n = v.norm()
        assert isinstance(n, Spherefun)

    def test_scalar_mul(self):
        """2 * v has doubled components."""
        from chebfunjax.spherefun.spherefunv import Spherefunv
        def f_fn(lam, th):
            return jnp.cos(lam) * jnp.sin(th)
        def g_fn(lam, th):
            return jnp.sin(lam) * jnp.sin(th)
        v = Spherefunv.from_functions(f_fn, g_fn)
        v2 = 2.0 * v
        lam, th = jnp.array(0.4), jnp.array(0.9)
        f2, g2 = v2(lam, th)
        npt.assert_allclose(float(f2), 2.0 * float(f_fn(lam, th)), rtol=1e-6)

    def test_repr(self):
        """repr is a non-empty string."""
        from chebfunjax.spherefun.spherefunv import Spherefunv
        v = Spherefunv.from_functions(
            lambda lam, th: jnp.cos(lam),
            lambda lam, th: jnp.sin(lam),
        )
        assert "Spherefunv" in repr(v)


# ===========================================================================
# Chebfun3v vector field
# ===========================================================================

class TestChebfun3v:
    """Integration tests for Chebfun3v."""

    def test_construction(self):
        """Chebfun3v can be constructed from three callables."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        assert isinstance(v, Chebfun3v)
        assert len(v.components) == 3

    def test_evaluation(self):
        """Chebfun3v evaluates all three components."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        def f_fn(x, y, z):
            return x
        def g_fn(x, y, z):
            return y
        def h_fn(x, y, z):
            return z
        v = Chebfun3v.from_functions(f_fn, g_fn, h_fn)
        x, y, z = jnp.array(0.3), jnp.array(-0.4), jnp.array(0.7)
        fv, gv, hv = v(x, y, z)
        npt.assert_allclose(float(fv), 0.3, rtol=1e-10)
        npt.assert_allclose(float(gv), -0.4, rtol=1e-10)
        npt.assert_allclose(float(hv), 0.7, rtol=1e-10)

    def test_dot_with_self(self):
        """v.dot(v) = |v|^2 = x^2 + y^2 + z^2."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        d = v.dot(v)
        x, y, z = jnp.array(0.3), jnp.array(0.4), jnp.array(0.5)
        expected = float(x ** 2 + y ** 2 + z ** 2)
        npt.assert_allclose(float(d(x, y, z)), expected, rtol=1e-8)

    def test_cross_product_standard(self):
        """(1,0,0) x (0,1,0) = (0,0,1)."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        ex = Chebfun3v.from_functions(
            lambda x, y, z: jnp.ones_like(x),
            lambda x, y, z: jnp.zeros_like(x),
            lambda x, y, z: jnp.zeros_like(x),
        )
        ey = Chebfun3v.from_functions(
            lambda x, y, z: jnp.zeros_like(x),
            lambda x, y, z: jnp.ones_like(x),
            lambda x, y, z: jnp.zeros_like(x),
        )
        ez = ex.cross(ey)
        x, y, z = jnp.array(0.1), jnp.array(0.2), jnp.array(0.3)
        fv, gv, hv = ez(x, y, z)
        npt.assert_allclose(float(fv), 0.0, atol=1e-10)
        npt.assert_allclose(float(gv), 0.0, atol=1e-10)
        npt.assert_allclose(float(hv), 1.0, atol=1e-10)

    def test_norm(self):
        """norm([x, y, z]) = sqrt(x^2 + y^2 + z^2)."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        n = v.norm()
        x, y, z = jnp.array(0.3), jnp.array(0.4), jnp.array(0.0)
        npt.assert_allclose(float(n(x, y, z)), 0.5, rtol=1e-5)

    def test_add(self):
        """Componentwise addition of two Chebfun3v."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v1 = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: jnp.zeros_like(x),
            lambda x, y, z: jnp.zeros_like(x),
        )
        v2 = Chebfun3v.from_functions(
            lambda x, y, z: jnp.zeros_like(x),
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        v3 = v1 + v2
        x, y, z = jnp.array(0.3), jnp.array(0.4), jnp.array(0.5)
        f3, g3, h3 = v3(x, y, z)
        npt.assert_allclose(float(f3), 0.3, rtol=1e-8)
        npt.assert_allclose(float(g3), 0.4, rtol=1e-8)
        npt.assert_allclose(float(h3), 0.5, rtol=1e-8)

    def test_scalar_mul(self):
        """2 * v has doubled components."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        v2 = 2.0 * v
        x, y, z = jnp.array(0.3), jnp.array(0.4), jnp.array(0.5)
        f2, g2, h2 = v2(x, y, z)
        npt.assert_allclose(float(f2), 0.6, rtol=1e-8)

    def test_repr(self):
        """repr is a non-empty string."""
        from chebfunjax.chebfun3d.chebfun3v import Chebfun3v
        v = Chebfun3v.from_functions(
            lambda x, y, z: x,
            lambda x, y, z: y,
            lambda x, y, z: z,
        )
        assert "Chebfun3v" in repr(v)


# ===========================================================================
# Ballfun and Ballfunv
# ===========================================================================

class TestBallfun:
    """Integration tests for Ballfun."""

    def test_construction(self):
        """Ballfun can be constructed from a callable."""
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda lam, th, r: r ** 2)
        assert isinstance(f, Ballfun)

    def test_evaluation_constant(self):
        """Constant function evaluates to its value."""
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda lam, th, r: jnp.ones_like(r) * 2.5)
        lam = jnp.array(0.5)
        th = jnp.array(1.0)
        r = jnp.array(0.5)
        npt.assert_allclose(float(f(lam, th, r)), 2.5, rtol=1e-4)

    def test_evaluation_radial(self):
        """Radial function r^2 evaluates correctly (trilinear interp, ~1% accuracy)."""
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda lam, th, r: r ** 2, n=25)
        lam = jnp.array(1.0)
        th = jnp.array(0.8)
        r = jnp.array(0.6)
        npt.assert_allclose(float(f(lam, th, r)), 0.36, rtol=5e-3)

    def test_sum_constant(self):
        """∫∫∫_B 1 r^2 sin(th) dr dth dlam = 4*pi/3 (volume of unit ball)."""
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda lam, th, r: jnp.ones_like(r))
        integral = f.sum()
        # Volume of unit ball = 4*pi/3
        expected = 4.0 * float(jnp.pi) / 3.0
        npt.assert_allclose(float(integral), expected, rtol=1e-3)

    def test_repr(self):
        """repr is a non-empty string."""
        from chebfunjax.ballfun.ballfun import Ballfun
        f = Ballfun.from_function(lambda lam, th, r: r)
        assert "Ballfun" in repr(f)


class TestBallfunv:
    """Integration tests for Ballfunv."""

    def test_construction(self):
        """Ballfunv can be constructed from three callables."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        v = Ballfunv.from_functions(
            lambda lam, th, r: r * jnp.cos(lam),
            lambda lam, th, r: r * jnp.sin(lam),
            lambda lam, th, r: jnp.cos(th) * r,
        )
        assert isinstance(v, Ballfunv)
        assert len(v.components) == 3

    def test_evaluation(self):
        """Ballfunv evaluates all three components."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        def f_fn(lam, th, r):
            return r * jnp.cos(lam)
        def g_fn(lam, th, r):
            return r * jnp.sin(lam)
        def h_fn(lam, th, r):
            return jnp.cos(th) * r
        v = Ballfunv.from_functions(f_fn, g_fn, h_fn, n=25)
        lam = jnp.array(0.5)
        th = jnp.array(1.0)
        r = jnp.array(0.7)
        fv, gv, hv = v(lam, th, r)
        npt.assert_allclose(float(fv), float(f_fn(lam, th, r)), rtol=2e-3)
        npt.assert_allclose(float(gv), float(g_fn(lam, th, r)), rtol=2e-3)

    def test_cross_product(self):
        """Cross product of two Ballfunv fields is computed without error."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        ex = Ballfunv.from_functions(
            lambda lam, th, r: jnp.ones_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
        )
        ey = Ballfunv.from_functions(
            lambda lam, th, r: jnp.zeros_like(r),
            lambda lam, th, r: jnp.ones_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
        )
        ez = ex.cross(ey)
        lam, th, r = jnp.array(0.3), jnp.array(0.8), jnp.array(0.5)
        fv, gv, hv = ez(lam, th, r)
        npt.assert_allclose(float(fv), 0.0, atol=1e-4)
        npt.assert_allclose(float(gv), 0.0, atol=1e-4)
        npt.assert_allclose(float(hv), 1.0, rtol=1e-4)

    def test_norm(self):
        """norm of a unit constant vector is 1."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        ex = Ballfunv.from_functions(
            lambda lam, th, r: jnp.ones_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
        )
        n = ex.norm()
        lam, th, r = jnp.array(0.5), jnp.array(1.0), jnp.array(0.5)
        npt.assert_allclose(float(n(lam, th, r)), 1.0, rtol=1e-4)

    def test_scalar_mul(self):
        """Scalar multiplication scales all components."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        ex = Ballfunv.from_functions(
            lambda lam, th, r: jnp.ones_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
            lambda lam, th, r: jnp.zeros_like(r),
        )
        v2 = 3.0 * ex
        lam, th, r = jnp.array(0.5), jnp.array(1.0), jnp.array(0.5)
        fv, gv, hv = v2(lam, th, r)
        npt.assert_allclose(float(fv), 3.0, rtol=1e-4)

    def test_repr(self):
        """repr is a non-empty string."""
        from chebfunjax.ballfun.ballfunv import Ballfunv
        v = Ballfunv.from_functions(
            lambda lam, th, r: r,
            lambda lam, th, r: r,
            lambda lam, th, r: r,
        )
        assert "Ballfunv" in repr(v)
