"""Tests for chebfunjax.utils.ratapprox — Padé, ratinterp, trigratinterp.

Tier 1: Unit tests (pure Python, no MATLAB).
Mathematical properties verified:
  - padeapprox: known (m, n) approximants to exp, geometric series
  - ratinterp: rational interpolation of 1/(x - pole)
  - trigratinterp: rational interpolation of 1/(sin(pi*x) - c)
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.ratapprox import padeapprox, ratinterp, trigratinterp

RTOL = 1e-10
ATOL = 1e-12


# ============================================================================
# Tier 1: padeapprox tests
# ============================================================================


class TestPadeApprox:
    """Tests for the robust SVD-based Padé approximant."""

    def test_pade_exp_returns_callable(self):
        """padeapprox(exp, 2, 2) should return a callable."""
        r_fn, a, b, mu, nu, poles, res = padeapprox(np.exp, 2, 2)
        assert callable(r_fn)

    def test_pade_exp_order(self):
        """Padé (2,2) of exp has exact degrees mu=2, nu=2."""
        _, a, b, mu, nu, _, _ = padeapprox(np.exp, 2, 2)
        assert mu == 2
        assert nu == 2

    def test_pade_exp_accuracy(self):
        """Padé (5,5) of exp should be accurate near 0."""
        r_fn, _, _, _, _, _, _ = padeapprox(np.exp, 5, 5)
        z = np.linspace(-0.5, 0.5, 50)
        npt.assert_allclose(r_fn(z), np.exp(z), rtol=1e-12)

    def test_pade_exp_accuracy_smaller(self):
        """Padé (3,2) of exp should be accurate near 0."""
        r_fn, _, _, mu, nu, _, _ = padeapprox(np.exp, 3, 2)
        # Padé (3,2) has 6 Taylor coefficients to match
        z = np.array([0.0, 0.1, 0.2, -0.1])
        npt.assert_allclose(r_fn(z), np.exp(z), rtol=1e-8)

    def test_pade_from_coeffs(self):
        """padeapprox with explicit Taylor coefficient vector."""
        # exp(z) = 1 + z + z^2/2 + z^3/6 + z^4/24 + ...
        c = np.array([1, 1, 0.5, 1 / 6, 1 / 24, 1 / 120, 1 / 720])
        r_fn, a, b, mu, nu, _, _ = padeapprox(c, 3, 3)
        assert callable(r_fn)
        z = 0.5
        # Padé [3,3] of exp has error O(z^7) ~ 0.5^7/5040 ~ 1e-6 relative
        npt.assert_allclose(r_fn(z), np.exp(z), rtol=1e-5)

    def test_pade_geometric_series(self):
        """Padé (1,1) of 1/(1-z) should give exact result."""
        # 1/(1-z) has Taylor coeffs [1, 1, 1, ...]
        # Padé (1,1): a=[1], b=[1,-1], r(z) = 1/(1-z) — exact
        c = np.ones(10)
        r_fn, a, b, mu, nu, _, _ = padeapprox(c, 1, 1)
        # Check at a point inside radius of convergence
        z = 0.5
        npt.assert_allclose(r_fn(z), 1.0 / (1.0 - z), rtol=1e-12)

    def test_pade_tol_zero_no_robustness(self):
        """tol=0 disables robustification; mu, nu should equal m, n."""
        _, _, _, mu, nu, _, _ = padeapprox(np.exp, 3, 3, tol=0.0)
        assert mu == 3
        assert nu == 3

    def test_pade_returns_seven_values(self):
        """padeapprox always returns a 7-tuple."""
        result = padeapprox(np.exp, 2, 2)
        assert len(result) == 7

    def test_pade_normalisation(self):
        """Denominator b should be normalized: b[0] = 1."""
        _, _, b, _, _, _, _ = padeapprox(np.exp, 3, 3)
        npt.assert_allclose(float(b[0]), 1.0, atol=1e-14)

    def test_pade_scalar_input(self):
        """padeapprox should evaluate correctly at a single point."""
        r_fn, _, _, _, _, _, _ = padeapprox(np.exp, 4, 4)
        val = r_fn(0.0)
        npt.assert_allclose(float(val), 1.0, atol=1e-14)


class TestPadeJaxSource:
    """Chebfun7574c77 padeapprox.m: zero degree, complex data and evaluation.

    Native CPU controls are retained in pade_jax_native_20261017. These
    independent contracts also require JAX arrays and differentiable evaluation.
    """

    @pytest.mark.parametrize("kind", ["none", "list", "jax_array"])
    @pytest.mark.parametrize("callable_input", [False, True])
    def test_empty_optional_arguments_follow_native_defaults(self, kind, callable_input):
        # padeapprox.m lines36-44: isempty selects defaults; zero is not empty.
        empty = None if kind == "none" else ([] if kind == "list" else jnp.empty((0,)))
        coefficients = jnp.array([1.0, 0.25j])
        f = (lambda z: 1 / (1 - 0.25j*z)) if callable_input else coefficients
        evaluate, a, b, mu, nu, poles, residues = padeapprox(
            f, 0, 1, tol=empty, r=empty, N_fft=empty
        )
        assert (mu, nu) == (0, 1)
        npt.assert_allclose(a, [1.0], rtol=1e-13, atol=0)
        npt.assert_allclose(b, [1.0, -0.25j], rtol=1e-13, atol=1e-15)
        assert poles.shape == residues.shape == (1,)
        z = jnp.array([0.0, 0.5 + 0.25j])
        npt.assert_allclose(evaluate(z), 1 / (1 - 0.25j*z), rtol=1e-13)

    def test_zero_degree_is_negative_infinity(self):
        evaluate, a, b, mu, nu, poles, residues = padeapprox([0.0], 0, 0)
        assert mu == -np.inf
        assert nu == 0
        npt.assert_array_equal(a, [0.0])
        npt.assert_array_equal(b, [1.0])
        assert poles.shape == residues.shape == (0,)
        npt.assert_array_equal(evaluate(jnp.array([0.0, 1.0j])), [0.0, 0.0])

    def test_tiny_complex_numerator_is_preserved(self):
        evaluate, a, b, mu, nu, _, _ = padeapprox([1e-20j, 1e-20j], 0, 1)
        assert (mu, nu) == (0, 1)
        npt.assert_allclose(a, [1e-20j], rtol=1e-13, atol=0)
        npt.assert_allclose(b, [1.0, -1.0], rtol=1e-13, atol=0)
        npt.assert_allclose(evaluate(0.5), 2e-20j, rtol=1e-13, atol=0)

    def test_fit_does_not_use_numpy_kernels(self, monkeypatch):
        from chebfunjax.utils import ratapprox

        class NoNumpy:
            def __getattr__(self, name):
                raise AssertionError(f"Padé called NumPy.{name}")

        monkeypatch.setattr(ratapprox, "np", NoNumpy())
        evaluate, a, b, _, _, poles, residues = padeapprox(
            lambda z: 1 / (1 - 0.25j * z), 0, 1, N_fft=64
        )
        assert all(isinstance(x, jax.Array) for x in (a, b, poles, residues))
        npt.assert_allclose(evaluate(0.25), 1 / (1 - 0.0625j), rtol=1e-13)

    def test_returned_handle_supports_jit_and_complex_jvp(self):
        evaluate, *_ = padeapprox([1.0, 1.0j], 0, 1)
        z = jnp.array([0.125 + 0.25j, -0.5j])
        direction = jnp.array([0.5j, 0.25 - 0.5j])
        value, tangent = jax.jvp(jax.jit(evaluate), (z,), (direction,))
        npt.assert_allclose(value, 1 / (1 - 1j * z), rtol=1e-13)
        npt.assert_allclose(tangent, 1j * direction / (1 - 1j * z)**2, rtol=1e-13)


# ============================================================================
# Tier 1: ratinterp tests
# ============================================================================


class TestRatInterp:
    """Tests for robust rational interpolation on Chebyshev points."""

    def test_ratinterp_returns_callable(self):
        """ratinterp should return a 7-tuple with a callable as first element."""
        result = ratinterp(lambda x: 1.0 / (x - 0.2), 5, 5)
        assert len(result) == 7
        assert callable(result[0])

    def test_ratinterp_pole_outside(self):
        """Rational interpolant to 1/(x - 2) on [-1,1] (pole outside domain)."""
        def f(x):
            return 1.0 / (x - 2.0)
        r_fn, a, b, mu, nu, poles, res = ratinterp(f, 8, 1)
        # Should be accurate on [-1, 1]
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-8)

    def test_ratinterp_reduces_resolved_function_degree(self):
        """A resolved (non-rational) function reduces the requested type.

        exp is entire, so a robustified type-(8, 8) interpolant collapses
        the denominator to a much lower degree (SVD rank detection): the
        exact value MATLAB reports here is nu = 4.  The approximant must
        still match exp to machine precision.
        """
        for grid in ("type1", "type2"):
            r_fn, a, b, mu, nu, poles, res = ratinterp(np.exp, 8, 8, xi=grid)
            assert nu == 4
            assert mu == 8
            xx = np.linspace(-1.0, 1.0, 50)
            npt.assert_allclose(r_fn(xx), np.exp(xx), atol=1e-12)

    def test_ratinterp_reduction_disabled_by_zero_tol(self):
        """With tol=0 robustness/reduction is off, so the full type is kept."""
        r_fn, a, b, mu, nu, poles, res = ratinterp(
            np.exp, 8, 8, xi="type2", tol=0.0)
        assert nu == 8

    def test_ratinterp_polynomial(self):
        """Rational interpolant to a polynomial should give zero denominator degree."""
        # x^3 is a polynomial; type (3, 0) should work
        def f(x):
            return x**3
        r_fn, a, b, mu, nu, poles, res = ratinterp(f, 3, 0)
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-10)

    def test_ratinterp_data_vector(self):
        """ratinterp with a data vector (not callable)."""
        from chebfunjax.utils.quadrature import chebpts
        n_pts = 21
        x = np.array(chebpts(n_pts, kind=2))
        f_vals = np.sin(x)
        r_fn, _, _, mu, nu, _, _ = ratinterp(f_vals, 10, 0)
        # For a smooth sin, the denominator should be trivial (nu=0)
        assert nu == 0
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), np.sin(x_test), rtol=1e-8)

    def test_ratinterp_type1_nodes(self):
        """ratinterp with type1 Chebyshev nodes."""
        def f(x):
            return 1.0 / (x - 1.5)
        r_fn, _, _, _, _, _, _ = ratinterp(f, 6, 2, xi="type1")
        x_test = np.linspace(-0.8, 0.8, 15)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-6)

    def test_ratinterp_equidistant_nodes(self):
        """ratinterp with equidistant nodes (polynomial case).

        Equidistant nodes give exact interpolation at the 9 nodes but have
        modest approximation accuracy (~1e-4) away from them — this is expected
        for degree-8 polynomial interpolation through 9 equidistant points.
        """
        def f(x):
            return np.cos(3 * x)
        r_fn, _, _, mu, nu, _, _ = ratinterp(f, 8, 0, xi="equidistant")
        # Exact interpolation at the 9 nodes
        xi_nodes = np.linspace(-1.0, 1.0, 9)
        npt.assert_allclose(r_fn(xi_nodes), f(xi_nodes), rtol=1e-12)
        # Reasonable accuracy in the interior (no Runge singularity here)
        x_test = np.linspace(-0.8, 0.8, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-3)

    def test_ratinterp_invalid_NN(self):
        """NN < m+n+1 should raise ValueError."""
        with pytest.raises(ValueError, match="NN"):
            ratinterp(lambda x: x, 5, 5, NN=5)

    def test_ratinterp_bad_xi_type(self):
        """Unknown xi string should raise ValueError."""
        with pytest.raises(ValueError):
            ratinterp(lambda x: x, 3, 3, xi="notavalidtype")

    def test_ratinterp_custom_domain(self):
        """ratinterp on a custom domain [0, 2]."""
        def f(x):
            return 1.0 / (x - 3.0)
        r_fn, _, _, _, _, _, _ = ratinterp(f, 6, 2, domain=(0.0, 2.0))
        x_test = np.linspace(0.1, 1.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-6)

    def test_ratinterp_tol_zero_no_robustness(self):
        """tol=0 disables robustification."""
        def f(x):
            return 1.0 / (x - 2.0)
        r_fn, _, _, mu, nu, _, _ = ratinterp(f, 5, 5, tol=0.0)
        # Should not crash
        assert callable(r_fn)

    def test_ratinterp_accuracy_high_degree(self):
        """Higher-degree rational interpolant to 1/(x-0.3) on [-1,1]."""
        def f(x):
            return 1.0 / (x - 0.3)
        # Use a generous approximation that captures the pole structure
        r_fn, a, b, mu, nu, _, _ = ratinterp(f, 8, 4)
        x_test = np.linspace(-0.9, 0.2, 15)  # avoid x near 0.3
        # Actual MATLAB public capture has mu=0,nu=1 and nonfinite r.
        # Retain this independent accuracy bound for the polynomial ratio.
        actual = (np.polynomial.chebyshev.chebval(x_test, np.asarray(a)) /
                  np.polynomial.chebyshev.chebval(x_test, np.asarray(b)))
        npt.assert_allclose(actual, f(x_test), rtol=1e-5)


# ============================================================================
# Tier 1: trigratinterp tests
# ============================================================================


class TestTrigRatInterp:
    """Tests for robust trigonometric rational interpolation."""

    def test_trigratinterp_returns_tuple(self):
        """trigratinterp returns a 7-tuple."""
        def f(x):
            return np.sin(np.pi * x) + 0.5
        result = trigratinterp(f, 3, 0)
        assert len(result) == 7

    def test_trigratinterp_callable_first(self):
        """First element of trigratinterp result is callable."""
        def f(x):
            return np.cos(2 * np.pi * x)
        r_fn, *_ = trigratinterp(f, 3, 0)
        assert callable(r_fn)

    def test_trigratinterp_trig_polynomial(self):
        """Trig interpolant to cos(pi*x) should be exact (type (1,0)).

        cos(pi*x) is the k=1 Fourier mode in the period-2 basis exp(i*pi*k*x),
        so it is a trig polynomial of degree 1 on [-1, 1].
        """
        def f(x):
            return np.cos(np.pi * x)
        r_fn, a, b, mu, nu, poles, res = trigratinterp(f, 1, 0)
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-8)

    def test_trigratinterp_sin_polynomial(self):
        """Trig interpolant to sin(pi*x) should be exact (type (1,0))."""
        def f(x):
            return np.sin(np.pi * x)
        r_fn, _, _, mu, nu, _, _ = trigratinterp(f, 1, 0)
        x_test = np.linspace(-0.8, 0.8, 15)
        # atol=1e-12 handles the case where sin(pi*x) = 0 at x=0
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-8, atol=1e-12)

    def test_trigratinterp_with_data_vector(self):
        """trigratinterp with a pre-sampled data vector."""
        N = 11
        x = np.linspace(-1, 1, N, endpoint=False)
        f_vals = np.cos(np.pi * x)
        r_fn, _, _, mu, nu, _, _ = trigratinterp(f_vals, 1, 0)
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), np.cos(np.pi * x_test), rtol=1e-6)

    def test_trigratinterp_invalid_NN(self):
        """NN < 2*(m+n)+1 should raise ValueError."""
        with pytest.raises(ValueError, match="NN"):
            trigratinterp(lambda x: x, 5, 5, NN=5)

    def test_trigratinterp_smooth_approx(self):
        """Type-(12,0) approximant to 1/(2-sin(pi*x)) should be accurate.

        This function has singularities at sin(pi*x) = 2, which lie in the
        complex plane.  A degree-12 trig polynomial gives error ~1e-7.
        """
        def f(x):
            return 1.0 / (2.0 - np.sin(np.pi * x))
        r_fn, _, _, mu, nu, _, _ = trigratinterp(f, 12, 0)
        x_test = np.linspace(-0.9, 0.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-6)

    def test_trigratinterp_custom_domain(self):
        """trigratinterp on custom domain [0, 2]."""
        def f(x):
            return np.cos(np.pi * x)
        r_fn, _, _, _, _, _, _ = trigratinterp(f, 2, 0, domain=(0.0, 2.0))
        x_test = np.linspace(0.1, 1.9, 20)
        npt.assert_allclose(r_fn(x_test), f(x_test), rtol=1e-6)
