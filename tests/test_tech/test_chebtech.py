"""Tests for chebfunjax.tech.chebtech — Chebtech2 core class.

JAX contract: evaluation is jit=yes, vmap=yes, grad=yes.
              adaptive construction is jit=NO (Python loop).

This is the vertical-slice test: if Chebtech2 works, the whole architecture
is validated.
"""


import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import Chebtech2, _clenshaw
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import coeffs2vals

# Tolerances from project.conf
RTOL = 1e-12
ATOL = 1e-14


# ============================================================================
# Tier 1: Mathematical properties (pure Python, no MATLAB)
# ============================================================================


class TestConstruction:
    """Tests for Chebtech2 construction methods."""

    def test_from_function_sin(self):
        """Adaptive construction of sin(x) should give ~14 coefficients."""
        f = Chebtech2.from_function(jnp.sin)
        assert f.ishappy
        assert f.n == 14

    def test_from_function_exp(self):
        """Adaptive construction of exp(x) should resolve."""
        f = Chebtech2.from_function(jnp.exp)
        assert f.ishappy
        assert f.n < 20

    def test_from_function_polynomial(self):
        """A degree-5 polynomial should be captured exactly in <= 6 coeffs."""
        def poly(x):
            return 3 * x**5 - 2 * x**3 + x - 1
        f = Chebtech2.from_function(poly)
        assert f.ishappy
        assert f.n <= 6

    def test_from_function_constant(self):
        """A constant function should give very few coefficients."""
        f = Chebtech2.from_function(lambda x: 0 * x + 3.14)
        assert f.ishappy
        assert f.n <= 2

    def test_from_function_fixed_n(self):
        """Fixed-length construction with specified n."""
        f = Chebtech2.from_function(jnp.sin, n=20)
        assert f.n == 20

    def test_from_function_fixed_n_values(self):
        """Fixed-length construction should interpolate correctly."""
        f = Chebtech2.from_function(jnp.sin, n=20)
        x = chebpts(20)
        npt.assert_allclose(np.array(f.values), np.array(jnp.sin(x)), atol=1e-15)

    def test_from_coeffs(self):
        """Construct from coefficients."""
        c = jnp.array([1.0, 0.5, -0.25], dtype=jnp.float64)
        f = Chebtech2.from_coeffs(c)
        npt.assert_allclose(np.array(f.coeffs), np.array(c), atol=1e-15)
        assert f.n == 3

    def test_from_values(self):
        """Construct from values at Chebyshev-2 points."""
        x = chebpts(5)
        v = jnp.sin(x)
        f = Chebtech2.from_values(v)
        assert f.n == 5
        # Values should round-trip
        npt.assert_allclose(np.array(f.values), np.array(v), rtol=1e-14)

    def test_from_values_roundtrip(self):
        """from_values recovers original coefficients after from_coeffs."""
        c = jnp.array([1.0, 0.5, -0.25, 0.1, 0.0], dtype=jnp.float64)
        f = Chebtech2.from_coeffs(c)
        g = Chebtech2.from_values(f.values)
        npt.assert_allclose(np.array(g.coeffs), np.array(c), rtol=1e-13, atol=1e-14)

    def test_from_function_unhappy(self):
        """A function that doesn't converge should set ishappy=False."""
        # Use a function that is very oscillatory
        with pytest.warns(UserWarning, match="did not converge"):
            f = Chebtech2.from_function(
                lambda x: jnp.sin(1000 * x), maxpow2=5
            )
        assert not f.ishappy


class TestEvaluation:
    """Tests for Chebtech2 evaluation via Clenshaw."""

    def test_sin_at_point(self):
        """Evaluate sin at x=0.5 should match jnp.sin(0.5)."""
        f = Chebtech2.from_function(jnp.sin)
        y = float(f(jnp.float64(0.5)))
        expected = float(jnp.sin(0.5))
        npt.assert_allclose(y, expected, rtol=RTOL, atol=ATOL)

    def test_sin_at_endpoints(self):
        """Evaluate at domain endpoints."""
        f = Chebtech2.from_function(jnp.sin)
        npt.assert_allclose(float(f(jnp.float64(-1.0))), float(jnp.sin(-1.0)),
                            rtol=1e-12)
        npt.assert_allclose(float(f(jnp.float64(1.0))), float(jnp.sin(1.0)),
                            rtol=1e-12)

    def test_sin_array(self):
        """Evaluate at an array of points."""
        f = Chebtech2.from_function(jnp.sin)
        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        ys = f(xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.sin(xs)), rtol=1e-12)

    def test_exp_at_points(self):
        """Evaluate exp at several points."""
        f = Chebtech2.from_function(jnp.exp)
        xs = jnp.array([-0.7, -0.3, 0.0, 0.2, 0.9], dtype=jnp.float64)
        ys = f(xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.exp(xs)), rtol=1e-13)

    def test_constant_eval(self):
        """Evaluate a constant Chebtech2."""
        f = Chebtech2.from_coeffs(jnp.array([3.14], dtype=jnp.float64))
        assert f.n == 1
        npt.assert_allclose(float(f(jnp.float64(0.5))), 3.14, atol=1e-15)
        npt.assert_allclose(float(f(jnp.float64(-1.0))), 3.14, atol=1e-15)

    def test_clenshaw_empty(self):
        """Empty coefficient array gives zero."""
        c = jnp.array([], dtype=jnp.float64)
        x = jnp.array([0.5], dtype=jnp.float64)
        y = _clenshaw(c, x)
        npt.assert_allclose(np.array(y), 0.0, atol=1e-15)

    def test_chebyshev_T3(self):
        """T_3(x) = 4x^3 - 3x. coeffs = [0, 0, 0, 1]."""
        c = jnp.array([0.0, 0.0, 0.0, 1.0], dtype=jnp.float64)
        f = Chebtech2.from_coeffs(c)
        xs = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=jnp.float64)
        expected = 4 * xs**3 - 3 * xs
        npt.assert_allclose(np.array(f(xs)), np.array(expected), atol=1e-14)


class TestProperties:
    """Tests for Chebtech2 properties."""

    def test_n(self):
        """n is the number of coefficients."""
        f = Chebtech2.from_function(jnp.sin)
        assert f.n == 14

    def test_len(self):
        """len(f) equals f.n."""
        f = Chebtech2.from_function(jnp.sin)
        assert len(f) == f.n

    def test_values(self):
        """values should match coeffs2vals(coeffs)."""
        f = Chebtech2.from_function(jnp.sin)
        v_expected = coeffs2vals(f.coeffs)
        npt.assert_allclose(np.array(f.values), np.array(v_expected), atol=1e-15)

    def test_vscale_sin(self):
        """vscale of sin should be close to 1 (max |sin(x)| on [-1,1] ~ 0.8415)."""
        f = Chebtech2.from_function(jnp.sin)
        npt.assert_allclose(f.vscale, float(jnp.sin(1.0)), rtol=1e-10)

    def test_vscale_exp(self):
        """vscale of exp should be close to exp(1) ~ 2.718."""
        f = Chebtech2.from_function(jnp.exp)
        npt.assert_allclose(f.vscale, float(jnp.exp(1.0)), rtol=1e-10)

    def test_repr(self):
        """repr should contain n and vscale."""
        f = Chebtech2.from_function(jnp.sin)
        s = repr(f)
        assert "Chebtech2" in s
        assert "n=14" in s
        assert "vscale=" in s

    def test_ishappy_true(self):
        """Adaptive construction of sin should be happy."""
        f = Chebtech2.from_function(jnp.sin)
        assert f.ishappy is True

    def test_ishappy_from_coeffs(self):
        """from_coeffs defaults to happy=True."""
        f = Chebtech2.from_coeffs(jnp.array([1.0, 0.5], dtype=jnp.float64))
        assert f.ishappy is True


class TestProlong:
    """Tests for prolong (zero-pad or truncate coefficients)."""

    def test_prolong_pad(self):
        """Prolonging to more coefficients zero-pads."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(20)
        assert g.n == 20
        npt.assert_allclose(np.array(g.coeffs[:f.n]), np.array(f.coeffs), atol=1e-15)
        npt.assert_allclose(np.array(g.coeffs[f.n:]), 0.0, atol=1e-15)

    def test_prolong_truncate(self):
        """Prolonging to fewer coefficients truncates."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(5)
        assert g.n == 5
        npt.assert_allclose(np.array(g.coeffs), np.array(f.coeffs[:5]), atol=1e-15)

    def test_prolong_same(self):
        """Prolonging to same length returns self."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(f.n)
        assert g is f

    def test_prolong_preserves_values(self):
        """Prolonged Chebtech2 evaluates the same at test points."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(30)
        xs = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)), rtol=1e-13)


class TestSimplify:
    """Tests for simplify (chop trailing coefficients)."""

    def test_simplify_padded(self):
        """Simplifying a padded representation should remove zeros."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(50)
        h = g.simplify()
        assert h.n <= f.n

    def test_simplify_idempotent(self):
        """Simplifying twice gives the same result."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.simplify()
        h = g.simplify()
        assert h.n == g.n

    def test_simplify_unhappy_noop(self):
        """Simplifying an unhappy Chebtech2 does nothing."""
        c = jnp.ones(50, dtype=jnp.float64)
        f = Chebtech2(coeffs=c, ishappy=False)
        g = f.simplify()
        assert g.n == f.n

    def test_simplify_preserves_accuracy(self):
        """Simplified representation should still evaluate accurately."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(100)
        h = g.simplify()
        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(h(xs)), np.array(jnp.sin(xs)), rtol=1e-12)


class TestVals2CoeffsStatic:
    """Tests for the static vals2coeffs / coeffs2vals methods on Chebtech2."""

    def test_roundtrip(self):
        """Roundtrip through vals2coeffs and coeffs2vals."""
        c = jnp.array([1.0, 0.5, -0.25, 0.1, 0.0], dtype=jnp.float64)
        v = Chebtech2.coeffs2vals(c)
        c_back = Chebtech2.vals2coeffs(v)
        npt.assert_allclose(np.array(c_back), np.array(c), rtol=1e-14, atol=1e-15)


# ============================================================================
# Tier 1: JAX transform tests
# ============================================================================


class TestJIT:
    """Tests that Chebtech2 evaluation works under JIT.

    JAX contract: evaluation is JIT-safe.
    """

    def test_jit_via_call(self):
        """f(x) is already JIT'd via @eqx.filter_jit on __call__."""
        f = Chebtech2.from_function(jnp.sin)
        y = f(jnp.float64(0.5))
        npt.assert_allclose(float(y), float(jnp.sin(0.5)), rtol=RTOL)

    def test_jit_lambda(self):
        """jax.jit(lambda x: f(x)) works."""
        f = Chebtech2.from_function(jnp.sin)
        fast = jax.jit(lambda x: f(x))
        y = fast(jnp.float64(0.5))
        npt.assert_allclose(float(y), float(jnp.sin(0.5)), rtol=RTOL)

    def test_jit_closure(self):
        """JIT-compiled closure over a Chebtech2 works."""
        f = Chebtech2.from_function(jnp.sin)

        @jax.jit
        def evaluate(x):
            return f(x)

        y = evaluate(jnp.float64(0.5))
        npt.assert_allclose(float(y), float(jnp.sin(0.5)), rtol=RTOL)

    def test_filter_jit_with_module_arg(self):
        """eqx.filter_jit passing the Module as an argument."""
        f = Chebtech2.from_function(jnp.sin)

        @eqx.filter_jit
        def evaluate(g, x):
            return g(x)

        y = evaluate(f, jnp.float64(0.5))
        npt.assert_allclose(float(y), float(jnp.sin(0.5)), rtol=RTOL)

    def test_jit_array_input(self):
        """JIT evaluation on an array of points."""
        f = Chebtech2.from_function(jnp.sin)
        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        fast = jax.jit(lambda x: f(x))
        ys = fast(xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.sin(xs)), rtol=1e-12)


class TestGrad:
    """Tests that Chebtech2 evaluation is differentiable.

    JAX contract: jax.grad(lambda x: f(x)) should give df/dx.
    """

    def test_grad_sin(self):
        """Gradient of sin at x=0.5 should be cos(0.5)."""
        f = Chebtech2.from_function(jnp.sin)
        df = jax.grad(lambda x: f(x))
        y = df(jnp.float64(0.5))
        expected = float(jnp.cos(0.5))
        npt.assert_allclose(float(y), expected, rtol=1e-10)

    def test_grad_exp(self):
        """Gradient of exp at x=0 should be exp(0)=1."""
        f = Chebtech2.from_function(jnp.exp)
        df = jax.grad(lambda x: f(x))
        y = df(jnp.float64(0.0))
        npt.assert_allclose(float(y), 1.0, rtol=1e-10)

    def test_grad_polynomial(self):
        """Gradient of x^3 at x=0.5 should be 3*0.5^2 = 0.75."""
        f = Chebtech2.from_function(lambda x: x**3)
        df = jax.grad(lambda x: f(x))
        y = df(jnp.float64(0.5))
        npt.assert_allclose(float(y), 0.75, rtol=1e-10)

    def test_grad_at_multiple_points(self):
        """Gradient of sin at several points matches cos."""
        f = Chebtech2.from_function(jnp.sin)
        df = jax.grad(lambda x: f(x))
        for x0 in [-0.9, -0.5, 0.0, 0.3, 0.8]:
            y = df(jnp.float64(x0))
            npt.assert_allclose(float(y), float(jnp.cos(x0)), rtol=1e-8)

    def test_grad_wrt_coefficients(self):
        """Gradient of f(x0) w.r.t. Chebyshev coefficients.

        d/dc_k [ sum_j c_j T_j(x0) ] = T_k(x0)
        """
        x0 = jnp.float64(0.3)
        n = 5

        def f_at_x0(c):
            g = Chebtech2.from_coeffs(c)
            return g(x0)

        c = jnp.array([1.0, 0.5, -0.25, 0.1, 0.0], dtype=jnp.float64)
        grad_c = jax.grad(f_at_x0)(c)

        # T_k(x0) = cos(k * arccos(x0))
        k = jnp.arange(n, dtype=jnp.float64)
        T_at_x0 = jnp.cos(k * jnp.arccos(x0))
        npt.assert_allclose(np.array(grad_c), np.array(T_at_x0), rtol=1e-12)


class TestVmap:
    """Tests that Chebtech2 evaluation works under vmap.

    JAX contract: jax.vmap(f) evaluates at many points in parallel.
    """

    def test_vmap_sin(self):
        """vmap evaluation of sin matches jnp.sin."""
        f = Chebtech2.from_function(jnp.sin)
        xs = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        ys = jax.vmap(lambda x: f(x))(xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.sin(xs)), rtol=1e-12)

    def test_vmap_exp(self):
        """vmap evaluation of exp."""
        f = Chebtech2.from_function(jnp.exp)
        xs = jnp.linspace(-1, 1, 30, dtype=jnp.float64)
        ys = jax.vmap(lambda x: f(x))(xs)
        npt.assert_allclose(np.array(ys), np.array(jnp.exp(xs)), rtol=1e-12)

    def test_vmap_matches_vectorized(self):
        """vmap and direct vectorized call give the same result."""
        f = Chebtech2.from_function(jnp.sin)
        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        ys_vmap = jax.vmap(lambda x: f(x))(xs)
        ys_direct = f(xs)
        npt.assert_allclose(np.array(ys_vmap), np.array(ys_direct), atol=1e-15)


# ============================================================================
# Tier 1: Immutability tests
# ============================================================================


class TestImmutability:
    """Chebtech2 is an equinox Module — all operations return new objects."""

    def test_prolong_returns_new(self):
        """prolong returns a different object (when n differs)."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(20)
        assert g is not f

    def test_simplify_returns_new(self):
        """simplify on padded Chebtech2 returns a different object."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.prolong(50)
        h = g.simplify()
        assert h is not g

    def test_module_is_pytree(self):
        """Chebtech2 is a valid JAX pytree."""
        f = Chebtech2.from_function(jnp.sin)
        leaves, treedef = jax.tree.flatten(f)
        # Should have exactly one leaf: coeffs
        assert len(leaves) == 1
        npt.assert_allclose(np.array(leaves[0]), np.array(f.coeffs), atol=1e-15)

        # Reconstruct
        g = jax.tree.unflatten(treedef, leaves)
        assert g.n == f.n


# ============================================================================
# Tier 1: Various test functions
# ============================================================================


class TestVariousFunctions:
    """Test adaptive construction on a range of smooth functions."""

    @pytest.mark.parametrize(
        "func, name, max_n",
        [
            (jnp.sin, "sin", 20),
            (jnp.cos, "cos", 20),
            (jnp.exp, "exp", 20),
            (lambda x: jnp.tanh(x), "tanh", 35),
            (lambda x: 1.0 / (1.0 + 25.0 * x**2), "runge", 200),
            (lambda x: jnp.exp(-x**2), "gaussian", 30),
            (lambda x: jnp.sin(10 * x), "sin(10x)", 50),
        ],
    )
    def test_adaptive_accuracy(self, func, name, max_n):
        """Adaptive construction resolves common test functions."""
        f = Chebtech2.from_function(func)
        assert f.ishappy, f"{name}: not happy (n={f.n})"
        assert f.n <= max_n, f"{name}: n={f.n} exceeds expected max {max_n}"

        # Check evaluation accuracy at random points
        rng = np.random.default_rng(42)
        xs = jnp.array(rng.uniform(-1, 1, 50), dtype=jnp.float64)
        ys = f(xs)
        expected = func(xs)
        npt.assert_allclose(
            np.array(ys), np.array(expected),
            rtol=1e-10, atol=1e-12,
            err_msg=f"{name}: evaluation failed",
        )


# ============================================================================
# Tier 2: MATLAB cross-validation
# ============================================================================


@pytest.mark.matlab
class TestMatlabCrossValidation:
    """Cross-validation against MATLAB Chebfun golden references.

    These tests load pre-computed MATLAB data and compare against our
    implementation. The golden refs are generated by matlab_harness/refs/chebtech.m.
    """

    @pytest.fixture(autouse=True)
    def _load_refs(self):
        """Load MATLAB reference data."""
        from tests.conftest import load_matlab_ref
        try:
            self.refs = load_matlab_ref("chebtech.mat")
        except Exception:
            pytest.skip("MATLAB reference chebtech.mat not available")

    def test_sin_coeffs(self):
        """Chebyshev coefficients of sin(x) match MATLAB."""
        f = Chebtech2.from_function(jnp.sin)
        ref_coeffs = self.refs["sin_coeffs"]
        n = min(f.n, len(ref_coeffs))
        npt.assert_allclose(
            np.array(f.coeffs[:n]), ref_coeffs[:n], rtol=1e-12, atol=1e-14
        )

    def test_sin_eval(self):
        """Evaluation of sin at test points matches MATLAB."""
        f = Chebtech2.from_function(jnp.sin)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        ref_vals = self.refs["sin_vals"]
        ys = f(test_pts)
        npt.assert_allclose(np.array(ys), ref_vals, rtol=1e-12, atol=1e-14)

    def test_exp_coeffs(self):
        """Chebyshev coefficients of exp(x) match MATLAB."""
        f = Chebtech2.from_function(jnp.exp)
        ref_coeffs = self.refs["exp_coeffs"]
        n = min(f.n, len(ref_coeffs))
        npt.assert_allclose(
            np.array(f.coeffs[:n]), ref_coeffs[:n], rtol=1e-12, atol=1e-14
        )

    def test_prolong_coeffs(self):
        """Prolonged coefficients match MATLAB prolong output."""
        f = Chebtech2.from_function(jnp.sin)
        ref_prolonged = self.refs["sin_prolonged_30"]
        g = f.prolong(30)
        npt.assert_allclose(
            np.array(g.coeffs), ref_prolonged, rtol=1e-14, atol=1e-15
        )
