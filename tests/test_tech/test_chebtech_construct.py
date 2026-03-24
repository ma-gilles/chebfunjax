"""Tests for Chebtech2 construction refinements: compose, restrict, happiness_check.

JAX contract: compose and restrict are NOT JIT-safe (adaptive construction).
              happiness_check is NOT JIT-safe (Python control flow).
              Evaluation of resulting Chebtech2 objects IS JIT/grad/vmap-safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.utils.quadrature import chebpts
from chebfunjax.utils.transforms import vals2coeffs

# Tolerances
RTOL = 1e-12
ATOL = 1e-14


# ============================================================================
# Tier 1: Mathematical properties (pure Python, no MATLAB)
# ============================================================================


class TestCompose:
    """Tests for Chebtech2.compose — apply an operator to a Chebtech2."""

    def test_compose_exp_sin(self):
        """compose(exp) applied to sin(x) should give exp(sin(x))."""
        sin_cheb = Chebtech2.from_function(jnp.sin)
        exp_sin = sin_cheb.compose(jnp.exp)
        assert exp_sin.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = jnp.exp(jnp.sin(xs))
        npt.assert_allclose(np.array(exp_sin(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_compose_cos_exp(self):
        """compose(cos) applied to exp(x) should give cos(exp(x))."""
        exp_cheb = Chebtech2.from_function(jnp.exp)
        cos_exp = exp_cheb.compose(jnp.cos)
        assert cos_exp.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = jnp.cos(jnp.exp(xs))
        npt.assert_allclose(np.array(cos_exp(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_compose_square(self):
        """compose(x^2) applied to sin(x) should give sin(x)^2."""
        sin_cheb = Chebtech2.from_function(jnp.sin)
        sin2 = sin_cheb.compose(lambda x: x**2)
        assert sin2.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = jnp.sin(xs) ** 2
        npt.assert_allclose(np.array(sin2(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_compose_identity(self):
        """compose(lambda x: x) should return essentially the same function."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.compose(lambda x: x)
        assert g.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        npt.assert_allclose(np.array(g(xs)), np.array(f(xs)),
                            rtol=1e-12, atol=1e-14)

    def test_compose_chebtech_objects(self):
        """compose(g) where g is a Chebtech2 computes g(f(x))."""
        f = Chebtech2.from_function(lambda x: x**2)
        g = Chebtech2.from_function(jnp.sin)
        # g(f(x)) = sin(x^2)
        h = f.compose(g)
        assert h.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = jnp.sin(xs**2)
        npt.assert_allclose(np.array(h(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_compose_binary_operator(self):
        """compose(op, g) with binary op computes op(f(x), g(x))."""
        f = Chebtech2.from_function(jnp.sin)
        g = Chebtech2.from_function(jnp.cos)
        # op(f(x), g(x)) = sin(x) + cos(x)
        h = f.compose(lambda a, b: a + b, g)
        assert h.ishappy

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        expected = jnp.sin(xs) + jnp.cos(xs)
        npt.assert_allclose(np.array(h(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_compose_returns_new_chebtech(self):
        """compose should return a new Chebtech2 object."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.compose(jnp.exp)
        assert g is not f
        assert isinstance(g, Chebtech2)

    def test_compose_harder_function(self):
        """compose with a more oscillatory function."""
        f = Chebtech2.from_function(lambda x: jnp.sin(5 * x))
        g = f.compose(jnp.exp)
        assert g.ishappy

        xs = jnp.linspace(-1, 1, 100, dtype=jnp.float64)
        expected = jnp.exp(jnp.sin(5 * xs))
        npt.assert_allclose(np.array(g(xs)), np.array(expected),
                            rtol=1e-10, atol=1e-12)

    def test_compose_result_is_jit_safe(self):
        """Evaluation of composed Chebtech2 should be JIT-safe."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.compose(jnp.exp)
        fast = jax.jit(lambda x: g(x))
        y = fast(jnp.float64(0.5))
        expected = float(jnp.exp(jnp.sin(0.5)))
        npt.assert_allclose(float(y), expected, rtol=1e-12)

    def test_compose_result_is_grad_safe(self):
        """Gradient through composed Chebtech2 evaluation should work."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.compose(jnp.exp)  # exp(sin(x))
        df = jax.grad(lambda x: g(x))
        # d/dx exp(sin(x)) = exp(sin(x)) * cos(x)
        x0 = jnp.float64(0.3)
        expected = float(jnp.exp(jnp.sin(x0)) * jnp.cos(x0))
        npt.assert_allclose(float(df(x0)), expected, rtol=1e-8)


class TestRestrict:
    """Tests for Chebtech2.restrict — restrict to a sub-interval."""

    def test_restrict_sin_to_0_05(self):
        """Restrict sin from [-1,1] to [0, 0.5]."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(0.0, 0.5)

        # g lives on [-1, 1] but represents sin on [0, 0.5]
        # x=-1 maps to 0, x=0 maps to 0.25, x=1 maps to 0.5
        npt.assert_allclose(float(g(jnp.float64(-1.0))),
                            float(jnp.sin(0.0)), atol=1e-14)
        npt.assert_allclose(float(g(jnp.float64(0.0))),
                            float(jnp.sin(0.25)), atol=1e-14)
        npt.assert_allclose(float(g(jnp.float64(1.0))),
                            float(jnp.sin(0.5)), atol=1e-14)

    def test_restrict_exp_to_m05_05(self):
        """Restrict exp from [-1,1] to [-0.5, 0.5]."""
        f = Chebtech2.from_function(jnp.exp)
        g = f.restrict(-0.5, 0.5)

        # x=-1 maps to -0.5, x=0 maps to 0, x=1 maps to 0.5
        npt.assert_allclose(float(g(jnp.float64(-1.0))),
                            float(jnp.exp(-0.5)), rtol=1e-13)
        npt.assert_allclose(float(g(jnp.float64(0.0))),
                            float(jnp.exp(0.0)), rtol=1e-13)
        npt.assert_allclose(float(g(jnp.float64(1.0))),
                            float(jnp.exp(0.5)), rtol=1e-13)

    def test_restrict_array_evaluation(self):
        """Restrict and evaluate at many points."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(0.0, 0.5)

        xs = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        # Map xs from [-1,1] to [0, 0.5]: y = 0.25*x + 0.25
        ys_mapped = 0.25 * xs + 0.25
        expected = jnp.sin(ys_mapped)
        npt.assert_allclose(np.array(g(xs)), np.array(expected),
                            rtol=1e-12, atol=1e-14)

    def test_restrict_full_interval(self):
        """Restrict to [-1, 1] should be a no-op (same coefficients)."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(-1.0, 1.0)
        npt.assert_allclose(np.array(g.coeffs), np.array(f.coeffs), atol=1e-14)

    def test_restrict_preserves_ishappy(self):
        """restrict preserves the ishappy flag."""
        f = Chebtech2.from_function(jnp.sin)
        assert f.ishappy
        g = f.restrict(0.0, 0.5)
        assert g.ishappy

    def test_restrict_returns_new_object(self):
        """restrict returns a new Chebtech2."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(-0.5, 0.5)
        assert g is not f
        assert isinstance(g, Chebtech2)

    def test_restrict_bad_interval(self):
        """restrict with invalid interval should raise ValueError."""
        f = Chebtech2.from_function(jnp.sin)
        with pytest.raises(ValueError, match="not a valid sub-interval"):
            f.restrict(-2.0, 0.5)
        with pytest.raises(ValueError, match="not a valid sub-interval"):
            f.restrict(0.5, 0.3)  # a >= b
        with pytest.raises(ValueError, match="not a valid sub-interval"):
            f.restrict(-0.5, 1.5)

    def test_restrict_narrow_interval(self):
        """Restrict to a very narrow interval."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(-0.01, 0.01)

        # x=0 maps to midpoint 0.0
        npt.assert_allclose(float(g(jnp.float64(0.0))),
                            float(jnp.sin(0.0)), atol=1e-14)
        # x=1 maps to 0.01
        npt.assert_allclose(float(g(jnp.float64(1.0))),
                            float(jnp.sin(0.01)), rtol=1e-12)

    def test_restrict_result_is_jit_safe(self):
        """Evaluation of restricted Chebtech2 should be JIT-safe."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(0.0, 0.5)
        fast = jax.jit(lambda x: g(x))
        # x=0 maps to 0.25
        y = fast(jnp.float64(0.0))
        npt.assert_allclose(float(y), float(jnp.sin(0.25)), atol=1e-14)

    def test_restrict_half_interval(self):
        """Restrict to right half [-1,1] -> [0,1], check several functions."""
        for func, name in [(jnp.sin, "sin"), (jnp.exp, "exp"),
                           (jnp.cos, "cos")]:
            f = Chebtech2.from_function(func)
            g = f.restrict(0.0, 1.0)
            xs = jnp.linspace(-1, 1, 30, dtype=jnp.float64)
            # Map xs from [-1,1] to [0,1]: y = 0.5*x + 0.5
            ys = 0.5 * xs + 0.5
            expected = func(ys)
            npt.assert_allclose(
                np.array(g(xs)), np.array(expected),
                rtol=1e-12, atol=1e-14,
                err_msg=f"restrict failed for {name}",
            )


class TestHappinessCheck:
    """Tests for the happiness_check static method."""

    def test_sin_33pts_is_happy(self):
        """Sin on a 33-point grid should be happy with cutoff ~14."""
        x = chebpts(33, kind=2)
        v = jnp.sin(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v, op=jnp.sin)
        assert ishappy is True
        assert cutoff == 14

    def test_exp_33pts_is_happy(self):
        """Exp on a 33-point grid should be happy with cutoff ~15."""
        x = chebpts(33, kind=2)
        v = jnp.exp(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v, op=jnp.exp)
        assert ishappy is True
        assert cutoff == 15

    def test_sin_17pts_not_happy(self):
        """Sin on a 17-point grid: standard_chop requires n >= 17 to chop."""
        x = chebpts(17, kind=2)
        v = jnp.sin(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v)
        # standard_chop needs more room than 17 coefficients for sin
        assert ishappy is False
        assert cutoff == 17

    def test_zero_function_is_happy(self):
        """The zero function should be happy with cutoff=1."""
        c = jnp.zeros(33, dtype=jnp.float64)
        v = jnp.zeros(33, dtype=jnp.float64)
        ishappy, cutoff = Chebtech2.happiness_check(c, v)
        assert ishappy is True
        assert cutoff == 1

    def test_constant_function_is_happy(self):
        """A constant function on a large enough grid should be happy."""
        n = 33
        v = jnp.full(n, 3.14, dtype=jnp.float64)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v)
        assert ishappy is True
        assert cutoff <= 3  # constant should need very few coefficients

    def test_sample_test_detects_bad_approximation(self):
        """If the interpolant is wrong at test points, sample test fails."""
        # Construct sin coefficients, then evaluate against a *different* op
        x = chebpts(33, kind=2)
        v = jnp.sin(x)
        c = vals2coeffs(v)
        # Pass op=cos, which is different from sin
        # The interpolant (sin) should NOT match cos at test points
        ishappy, cutoff = Chebtech2.happiness_check(c, v, op=jnp.cos)
        # The standard_chop part will say "happy", but the sample test
        # should detect the mismatch and set ishappy=False
        assert ishappy is False

    def test_happiness_check_no_op(self):
        """Without an operator, only standard_chop is checked (no sample test)."""
        x = chebpts(33, kind=2)
        v = jnp.sin(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v)
        # Without op, no sample test — just standard_chop
        assert ishappy is True
        assert cutoff == 14

    def test_happiness_matches_construction(self):
        """The happiness check cutoff should match adaptive construction length."""
        f = Chebtech2.from_function(jnp.sin)
        assert f.n == 14  # matches MATLAB

        f2 = Chebtech2.from_function(jnp.exp)
        assert f2.n == 15  # matches MATLAB


# ============================================================================
# Tier 1: Various compose test functions
# ============================================================================


class TestComposeVariousFunctions:
    """Test compose on a range of function combinations."""

    @pytest.mark.parametrize(
        "inner, outer, name, max_err",
        [
            (jnp.sin, jnp.exp, "exp(sin)", 1e-12),
            (jnp.exp, jnp.cos, "cos(exp)", 1e-12),
            (jnp.cos, lambda x: x**2, "cos^2", 1e-12),
            (lambda x: jnp.tanh(x), jnp.exp, "exp(tanh)", 1e-12),
            (jnp.sin, lambda x: 1.0 / (1.0 + x**2), "1/(1+sin^2)", 1e-11),
        ],
    )
    def test_compose_accuracy(self, inner, outer, name, max_err):
        """Composed Chebtech2 evaluates accurately."""
        f = Chebtech2.from_function(inner)
        g = f.compose(outer)
        assert g.ishappy, f"{name}: not happy (n={g.n})"

        xs = jnp.linspace(-1, 1, 100, dtype=jnp.float64)
        expected = outer(inner(xs))
        err = float(jnp.max(jnp.abs(g(xs) - expected)))
        assert err < max_err, f"{name}: error {err:.2e} exceeds {max_err:.2e}"


# ============================================================================
# Tier 2: MATLAB cross-validation
# ============================================================================


@pytest.mark.matlab
class TestMatlabCrossValidation:
    """Cross-validation against MATLAB Chebfun golden references.

    These tests load pre-computed MATLAB data from
    tests/references/chebtech_construct.mat.
    """

    @pytest.fixture(autouse=True)
    def _load_refs(self):
        """Load MATLAB reference data."""
        from tests.conftest import load_matlab_ref
        try:
            self.refs = load_matlab_ref("chebtech_construct.mat")
        except Exception:
            pytest.skip("MATLAB reference chebtech_construct.mat not available")

    def test_compose_exp_sin_accuracy(self):
        """exp(sin(x)) evaluation matches MATLAB at test points."""
        sin_cheb = Chebtech2.from_function(jnp.sin)
        exp_sin = sin_cheb.compose(jnp.exp)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        our_vals = np.array(exp_sin(test_pts))
        exact = self.refs["compose_exp_sin_exact"]
        npt.assert_allclose(our_vals, exact, rtol=1e-12, atol=1e-14)

    def test_compose_exp_sin_n(self):
        """exp(sin(x)) should need same number of coefficients as MATLAB."""
        sin_cheb = Chebtech2.from_function(jnp.sin)
        exp_sin = sin_cheb.compose(jnp.exp)
        matlab_n = int(self.refs["compose_exp_sin_n"])
        assert exp_sin.n == matlab_n

    def test_compose_cos_exp_accuracy(self):
        """cos(exp(x)) evaluation matches MATLAB at test points."""
        exp_cheb = Chebtech2.from_function(jnp.exp)
        cos_exp = exp_cheb.compose(jnp.cos)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        our_vals = np.array(cos_exp(test_pts))
        exact = self.refs["compose_cos_exp_exact"]
        npt.assert_allclose(our_vals, exact, rtol=1e-12, atol=1e-14)

    def test_compose_sin_x2_accuracy(self):
        """sin(x^2) via compose matches MATLAB."""
        f = Chebtech2.from_function(lambda x: x**2)
        g = Chebtech2.from_function(jnp.sin)
        h = f.compose(g)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        our_vals = np.array(h(test_pts))
        exact = self.refs["compose_sin_x2_exact"]
        npt.assert_allclose(our_vals, exact, rtol=1e-12, atol=1e-14)

    def test_restrict_sin_0_05_accuracy(self):
        """sin restricted to [0, 0.5] matches MATLAB evaluation."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(0.0, 0.5)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        our_vals = np.array(g(test_pts))
        exact = self.refs["restrict_sin_0_05_exact"]
        npt.assert_allclose(our_vals, exact, rtol=1e-12, atol=1e-14)

    def test_restrict_sin_0_05_n(self):
        """Restricted sin should have same n as MATLAB."""
        f = Chebtech2.from_function(jnp.sin)
        g = f.restrict(0.0, 0.5)
        matlab_n = int(self.refs["restrict_sin_0_05_n"])
        assert g.n == matlab_n

    def test_restrict_exp_m05_05_accuracy(self):
        """exp restricted to [-0.5, 0.5] matches MATLAB."""
        f = Chebtech2.from_function(jnp.exp)
        g = f.restrict(-0.5, 0.5)
        test_pts = jnp.array(self.refs["test_pts"], dtype=jnp.float64)
        our_vals = np.array(g(test_pts))
        exact = self.refs["restrict_exp_m05_05_exact"]
        npt.assert_allclose(our_vals, exact, rtol=1e-12, atol=1e-14)

    def test_happiness_sin_33(self):
        """Happiness check for sin with 33 points matches MATLAB."""
        x = chebpts(33, kind=2)
        v = jnp.sin(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v, op=jnp.sin)
        assert ishappy == bool(self.refs["happy_sin33"])
        assert cutoff == int(self.refs["cutoff_sin33"])

    def test_happiness_exp_33(self):
        """Happiness check for exp with 33 points matches MATLAB."""
        x = chebpts(33, kind=2)
        v = jnp.exp(x)
        c = vals2coeffs(v)
        ishappy, cutoff = Chebtech2.happiness_check(c, v, op=jnp.exp)
        assert ishappy == bool(self.refs["happy_exp33"])
        assert cutoff == int(self.refs["cutoff_exp33"])
