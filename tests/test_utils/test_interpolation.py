"""Tests for chebfunjax.utils.interpolation — barycentric interpolation utilities.

JAX contract: bary, bary_weights, trig_bary, trig_bary_weights, barymat are all
JIT-safe. cheb_bary_weights uses Python control flow on n (static).
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.interpolation import (
    bary,
    bary_weights,
    barymat,
    cheb_bary_weights,
    trig_bary,
    trig_bary_weights,
)
from chebfunjax.utils.quadrature import chebpts


# ---------------------------------------------------------------------------
# Tier 1: cheb_bary_weights
# ---------------------------------------------------------------------------

class TestChebBaryWeights:
    """Tests for cheb_bary_weights.

    JAX contract: jit=partial (n must be static), vmap=no, grad=no
    """

    def test_n0(self):
        v = cheb_bary_weights(0)
        assert len(v) == 0

    def test_n1(self):
        v = cheb_bary_weights(1)
        npt.assert_allclose(np.array(v), [1.0], atol=1e-15)

    def test_n2(self):
        """n=2: [-0.5, 0.5]"""
        v = cheb_bary_weights(2)
        npt.assert_allclose(np.array(v), [-0.5, 0.5], atol=1e-15)

    def test_n3(self):
        """n=3: [0.5, -1, 0.5]"""
        v = cheb_bary_weights(3)
        npt.assert_allclose(np.array(v), [0.5, -1.0, 0.5], atol=1e-15)

    def test_n5(self):
        """n=5: [0.5, -1, 1, -1, 0.5]"""
        v = cheb_bary_weights(5)
        npt.assert_allclose(np.array(v), [0.5, -1.0, 1.0, -1.0, 0.5], atol=1e-15)

    def test_last_entry_positive(self):
        """The last entry should always be positive."""
        for n in range(2, 50):
            v = cheb_bary_weights(n)
            assert float(v[-1]) > 0, f"Last entry not positive for n={n}"

    def test_inf_norm_one(self):
        """max(|v|) == 1 for n >= 3 (n=2 gives max=0.5 by convention)."""
        for n in [3, 5, 10, 32, 64]:
            v = cheb_bary_weights(n)
            npt.assert_allclose(float(jnp.max(jnp.abs(v))), 1.0, atol=1e-15)

    def test_n2_endpoint_weights(self):
        """n=2: both entries are halved, so max(|v|) == 0.5."""
        v = cheb_bary_weights(2)
        npt.assert_allclose(float(jnp.max(jnp.abs(v))), 0.5, atol=1e-15)

    def test_alternating_sign(self):
        """Interior entries should alternate in sign."""
        for n in [5, 10, 20]:
            v = cheb_bary_weights(n)
            # Check that consecutive entries have opposite signs
            signs = np.sign(np.array(v))
            for i in range(n - 1):
                assert signs[i] * signs[i + 1] < 0, (
                    f"Non-alternating at positions {i},{i+1} for n={n}"
                )

    @pytest.mark.matlab
    def test_vs_matlab(self, matlab_interpolation):
        """Compare against MATLAB chebtech2.barywts."""
        for n in [2, 3, 5, 10, 17, 32, 64, 128]:
            v = cheb_bary_weights(n)
            ref = matlab_interpolation[f"cheb_barywts_n{n}"]
            npt.assert_allclose(np.array(v), ref, rtol=1e-14, atol=1e-15,
                                err_msg=f"n={n}")


# ---------------------------------------------------------------------------
# Tier 1: bary_weights (general)
# ---------------------------------------------------------------------------

class TestBaryWeights:
    """Tests for bary_weights.

    JAX contract: jit=yes, vmap=yes, grad=yes
    """

    def test_chebyshev_proportional(self):
        """For Chebyshev nodes, general weights should be proportional to
        the explicit Chebyshev weights."""
        for n in [5, 10, 20]:
            xk = chebpts(n, kind=2)
            w_gen = bary_weights(xk)
            w_cheb = cheb_bary_weights(n)
            # Normalise both to inf-norm 1 (they should already be)
            ratio = np.array(w_gen) / np.array(w_cheb)
            # All entries should have the same absolute ratio
            npt.assert_allclose(np.abs(ratio), np.abs(ratio[0]),
                                rtol=1e-12, atol=1e-14,
                                err_msg=f"n={n}")

    def test_inf_norm_one(self):
        """Weights are scaled so max(|w|) == 1."""
        xk = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=jnp.float64)
        w = bary_weights(xk)
        npt.assert_allclose(float(jnp.max(jnp.abs(w))), 1.0, atol=1e-14)

    def test_jit(self):
        """bary_weights is JIT-compatible."""
        xk = chebpts(10, kind=2)
        w_nojit = bary_weights(xk)
        w_jit = jax.jit(bary_weights)(xk)
        npt.assert_allclose(np.array(w_jit), np.array(w_nojit), rtol=1e-15)

    @pytest.mark.matlab
    def test_vs_matlab_cheb(self, matlab_interpolation):
        """Compare general bary_weights on Chebyshev nodes vs MATLAB."""
        for n in [5, 10, 17, 32]:
            xk = chebpts(n, kind=2)
            w = bary_weights(xk)
            ref = matlab_interpolation[f"bary_weights_cheb_n{n}"]
            npt.assert_allclose(np.array(w), ref, rtol=1e-11, atol=1e-13,
                                err_msg=f"n={n}")

    @pytest.mark.matlab
    def test_vs_matlab_random(self, matlab_interpolation):
        """Compare bary_weights on random nodes vs MATLAB."""
        xk = jnp.array(matlab_interpolation["bary_weights_rand10_nodes"],
                        dtype=jnp.float64)
        w = bary_weights(xk)
        ref = matlab_interpolation["bary_weights_rand10"]
        npt.assert_allclose(np.array(w), ref, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------
# Tier 1: bary
# ---------------------------------------------------------------------------

class TestBary:
    """Tests for bary (polynomial barycentric interpolation).

    JAX contract: jit=yes, vmap=yes, grad=yes
    """

    def test_polynomial_exactness(self):
        """Interpolation of polynomial of degree < n should be exact."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        # p(x) = 3x^5 - 2x^3 + x - 7
        fk = 3 * xk**5 - 2 * xk**3 + xk - 7
        x = jnp.linspace(-1, 1, 100, dtype=jnp.float64)
        fx = bary(x, fk, xk, vk)
        exact = 3 * x**5 - 2 * x**3 + x - 7
        npt.assert_allclose(np.array(fx), np.array(exact), rtol=1e-13, atol=1e-13)

    def test_constant_function(self):
        """Constant interpolation."""
        for n in [1, 5, 10]:
            xk = chebpts(max(n, 2), kind=2) if n > 1 else jnp.array([0.0])
            vk = cheb_bary_weights(max(n, 2)) if n > 1 else jnp.array([1.0])
            fk = 42.0 * jnp.ones_like(xk)
            x = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
            fx = bary(x, fk, xk, vk)
            npt.assert_allclose(np.array(fx), 42.0, rtol=1e-14)

    def test_runge_convergence(self):
        """Interpolation of the Runge function should converge exponentially."""
        runge = lambda x: 1.0 / (1.0 + 25.0 * x**2)
        x = jnp.linspace(-1, 1, 500, dtype=jnp.float64)
        exact = runge(x)

        prev_err = 1.0
        for n in [10, 20, 40, 80]:
            xk = chebpts(n, kind=2)
            vk = cheb_bary_weights(n)
            fk = runge(xk)
            fx = bary(x, fk, xk, vk)
            err = float(jnp.max(jnp.abs(fx - exact)))
            assert err < prev_err, (
                f"Convergence stalled at n={n}: err={err:.2e} >= prev={prev_err:.2e}"
            )
            prev_err = err

        # At n=80, Runge function should be well-resolved (geometric convergence).
        # The Runge function 1/(1+25x^2) has a singularity at x = +/-i/5 in the
        # complex plane, giving convergence rate rho = (1/5 + sqrt(1+1/25))^(-1).
        # At n=80 we expect about 7 digits; at n=160 about 14 digits.
        assert prev_err < 1e-5, f"Runge not converged at n=80: {prev_err:.2e}"

    def test_evaluation_at_nodes(self):
        """Evaluating at the interpolation nodes should return the function values."""
        n = 15
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.exp(xk)
        fx = bary(xk, fk, xk, vk)
        npt.assert_allclose(np.array(fx), np.array(fk), rtol=1e-14, atol=1e-15)

    def test_single_point(self):
        """Evaluate at a single point."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.sin(xk)
        x = jnp.array([0.5], dtype=jnp.float64)
        fx = bary(x, fk, xk, vk)
        npt.assert_allclose(float(fx[0]), np.sin(0.5), rtol=1e-13)

    def test_jit(self):
        """bary is JIT-compatible."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.sin(xk)
        x = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        fx_nojit = bary(x, fk, xk, vk)
        fx_jit = jax.jit(bary)(x, fk, xk, vk)
        npt.assert_allclose(np.array(fx_jit), np.array(fx_nojit), rtol=1e-15)

    def test_grad(self):
        """bary supports JAX grad (derivative of interpolant)."""
        n = 20
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.sin(xk)

        # Derivative of the interpolant at x=0.3 should approximate cos(0.3)
        def f_at_point(x0):
            return bary(x0[None], fk, xk, vk)[0]

        x0 = jnp.array(0.3, dtype=jnp.float64)
        grad_val = jax.grad(f_at_point)(x0)
        npt.assert_allclose(float(grad_val), np.cos(0.3), rtol=1e-10)

    @pytest.mark.matlab
    def test_runge_vs_matlab(self, matlab_interpolation):
        """Compare Runge function interpolation against MATLAB bary."""
        for n in [5, 10, 20, 50]:
            xk = chebpts(n, kind=2)
            vk = cheb_bary_weights(n)
            fk = 1.0 / (1.0 + 25.0 * xk**2)
            xx = jnp.array(matlab_interpolation[f"bary_runge_n{n}_xx"],
                           dtype=jnp.float64)
            fx = bary(xx, fk, xk, vk)
            ref = matlab_interpolation[f"bary_runge_n{n}_fx"]
            npt.assert_allclose(np.array(fx), ref, rtol=1e-12, atol=1e-14,
                                err_msg=f"n={n}")

    @pytest.mark.matlab
    def test_polynomial_vs_matlab(self, matlab_interpolation):
        """Compare x^4 interpolation against MATLAB bary."""
        n = 5
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = xk**4
        xx = jnp.array(matlab_interpolation["bary_x4_n5_xx"], dtype=jnp.float64)
        fx = bary(xx, fk, xk, vk)
        ref = matlab_interpolation["bary_x4_n5_fx"]
        npt.assert_allclose(np.array(fx), ref, rtol=1e-12, atol=1e-14)

    @pytest.mark.matlab
    def test_at_nodes_vs_matlab(self, matlab_interpolation):
        """Evaluating at nodes should match MATLAB."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.sin(xk)
        fx = bary(xk, fk, xk, vk)
        ref = matlab_interpolation["bary_at_nodes_n10"]
        npt.assert_allclose(np.array(fx), ref, rtol=1e-14, atol=1e-15)


# ---------------------------------------------------------------------------
# Tier 1: trig_bary_weights
# ---------------------------------------------------------------------------

class TestTrigBaryWeights:
    """Tests for trig_bary_weights.

    JAX contract: jit=yes, vmap=yes, grad=no
    """

    def test_equispaced_alternating(self):
        """For equispaced nodes, weights should be alternating +/-1 (up to global sign)."""
        for n in [4, 8, 16]:
            xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
            w = trig_bary_weights(xk)
            # Absolute values should be 1
            npt.assert_allclose(np.abs(np.array(w)), 1.0, atol=1e-12,
                                err_msg=f"n={n}")
            # Should be alternating
            signs = np.sign(np.array(w))
            npt.assert_allclose(np.abs(np.diff(signs)), 2.0, atol=1e-14,
                                err_msg=f"n={n}: not alternating")

    def test_inf_norm_one(self):
        """Weights should be normalised to max(|w|) == 1."""
        xk = jnp.array([-2.5, -1.2, 0.1, 0.8, 2.3], dtype=jnp.float64)
        w = trig_bary_weights(xk)
        npt.assert_allclose(float(jnp.max(jnp.abs(w))), 1.0, atol=1e-14)

    def test_jit(self):
        """trig_bary_weights is JIT-compatible."""
        xk = jnp.linspace(-jnp.pi, jnp.pi, 8, endpoint=False, dtype=jnp.float64)
        w_nojit = trig_bary_weights(xk)
        w_jit = jax.jit(trig_bary_weights)(xk)
        npt.assert_allclose(np.array(w_jit), np.array(w_nojit), rtol=1e-15)

    @pytest.mark.matlab
    def test_equispaced_vs_matlab(self, matlab_interpolation):
        """Compare equispaced trig weights vs MATLAB."""
        for n in [4, 8, 16, 32]:
            xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
            w = trig_bary_weights(xk)
            ref = matlab_interpolation[f"trig_barywts_eq_n{n}"]
            # Allow global sign difference: compare |w| and alternation
            npt.assert_allclose(np.abs(np.array(w)), np.abs(ref),
                                rtol=1e-12, atol=1e-13, err_msg=f"n={n}")

    @pytest.mark.matlab
    def test_nonuniform_vs_matlab(self, matlab_interpolation):
        """Compare non-equispaced trig weights vs MATLAB."""
        xk = jnp.array(matlab_interpolation["trig_barywts_nonunif_nodes"],
                        dtype=jnp.float64)
        w = trig_bary_weights(xk)
        ref = matlab_interpolation["trig_barywts_nonunif"]
        npt.assert_allclose(np.array(w), ref, rtol=1e-11, atol=1e-13)


# ---------------------------------------------------------------------------
# Tier 1: trig_bary
# ---------------------------------------------------------------------------

class TestTrigBary:
    """Tests for trig_bary (trigonometric barycentric interpolation).

    JAX contract: jit=yes (dom must be explicit), vmap=yes, grad=yes
    """

    def test_sin_exact(self):
        """sin(x) on enough equispaced points should be exact."""
        n = 16
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.sin(xk)
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        x = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 100, dtype=jnp.float64)
        fx = trig_bary(x, fk, xk, dom)
        npt.assert_allclose(np.array(fx), np.sin(np.array(x)), rtol=1e-13, atol=1e-14)

    def test_cos3x_exact(self):
        """cos(3x) on 16 equispaced points should be very accurate."""
        n = 16
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.cos(3.0 * xk)
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        x = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 100, dtype=jnp.float64)
        fx = trig_bary(x, fk, xk, dom)
        npt.assert_allclose(np.array(fx), np.cos(3.0 * np.array(x)),
                            rtol=1e-12, atol=1e-13)

    def test_convergence(self):
        """Trigonometric interpolation should converge for smooth periodic functions."""
        f = lambda x: jnp.exp(jnp.sin(x))
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        x = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 200, dtype=jnp.float64)
        exact = f(x)

        errors = []
        for n in [8, 16, 32, 64]:
            xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
            fk = f(xk)
            fx = trig_bary(x, fk, xk, dom)
            err = float(jnp.max(jnp.abs(fx - exact)))
            errors.append(err)

        # Geometric convergence: each doubling should reduce error significantly
        # until we hit machine epsilon
        for i in range(len(errors) - 1):
            if errors[i] > 1e-13:  # above machine epsilon
                assert errors[i + 1] < errors[i], (
                    f"Convergence stalled: n={[8,16,32,64][i+1]}, "
                    f"err={errors[i+1]:.2e} >= prev={errors[i]:.2e}"
                )
        # Should reach machine precision at n=64
        assert errors[-1] < 1e-13, f"Not converged at n=64: {errors[-1]:.2e}"

    def test_odd_n(self):
        """Works correctly with odd number of nodes."""
        n = 15
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.sin(2.0 * xk) + jnp.cos(3.0 * xk)
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        x = jnp.linspace(-jnp.pi + 0.01, jnp.pi - 0.01, 50, dtype=jnp.float64)
        fx = trig_bary(x, fk, xk, dom)
        exact = jnp.sin(2.0 * x) + jnp.cos(3.0 * x)
        npt.assert_allclose(np.array(fx), np.array(exact), rtol=1e-12, atol=1e-13)

    def test_default_domain(self):
        """Default domain should be [-pi, pi]."""
        n = 16
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.sin(xk)
        x = jnp.linspace(-jnp.pi + 0.1, jnp.pi - 0.1, 20, dtype=jnp.float64)
        fx1 = trig_bary(x, fk, xk)  # default
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        fx2 = trig_bary(x, fk, xk, dom)  # explicit
        npt.assert_allclose(np.array(fx1), np.array(fx2), rtol=1e-15)

    def test_jit(self):
        """trig_bary under user JIT."""
        n = 16
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.sin(xk)
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        x = jnp.linspace(-jnp.pi + 0.1, jnp.pi - 0.1, 20, dtype=jnp.float64)

        @jax.jit
        def eval_fn(x, fk, xk, dom):
            return trig_bary(x, fk, xk, dom)

        fx_jit = eval_fn(x, fk, xk, dom)
        fx_nojit = trig_bary(x, fk, xk, dom)
        npt.assert_allclose(np.array(fx_jit), np.array(fx_nojit), rtol=1e-15)

    @pytest.mark.matlab
    def test_sin_vs_matlab(self, matlab_interpolation):
        """Compare sin interpolation vs MATLAB trigBary."""
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        for n in [8, 16, 32]:
            xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
            fk = jnp.sin(xk)
            xx = jnp.array(matlab_interpolation[f"trigbary_sin_n{n}_xx"],
                           dtype=jnp.float64)
            fx = trig_bary(xx, fk, xk, dom)
            ref = matlab_interpolation[f"trigbary_sin_n{n}_fx"]
            npt.assert_allclose(np.array(fx), ref, rtol=1e-12, atol=1e-13,
                                err_msg=f"n={n}")

    @pytest.mark.matlab
    def test_cos3x_vs_matlab(self, matlab_interpolation):
        """Compare cos(3x) interpolation vs MATLAB trigBary."""
        n = 16
        dom = jnp.array([-jnp.pi, jnp.pi], dtype=jnp.float64)
        xk = jnp.linspace(-jnp.pi, jnp.pi, n, endpoint=False, dtype=jnp.float64)
        fk = jnp.cos(3.0 * xk)
        xx = jnp.array(matlab_interpolation["trigbary_cos3x_n16_xx"],
                       dtype=jnp.float64)
        fx = trig_bary(xx, fk, xk, dom)
        ref = matlab_interpolation["trigbary_cos3x_n16_fx"]
        npt.assert_allclose(np.array(fx), ref, rtol=1e-12, atol=1e-13)


# ---------------------------------------------------------------------------
# Tier 1: barymat
# ---------------------------------------------------------------------------

class TestBarymat:
    """Tests for barymat (barycentric interpolation matrix).

    JAX contract: jit=yes (with explicit w), vmap=no, grad=yes
    """

    def test_consistency_with_bary(self):
        """B @ f should match bary(y, f, x, w)."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = jnp.sin(xk)
        y = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)

        B = barymat(y, xk, vk)
        fx_mat = B @ fk
        fx_bary = bary(y, fk, xk, vk)
        npt.assert_allclose(np.array(fx_mat), np.array(fx_bary),
                            rtol=1e-13, atol=1e-14)

    def test_identity_when_grids_match(self):
        """When y == x, the matrix should be the identity."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        B = barymat(xk, xk, vk)
        npt.assert_allclose(np.array(B), np.eye(n), atol=1e-14)

    def test_default_weights(self):
        """Default weights (Chebyshev 2nd kind) should work."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        B_explicit = barymat(y, xk, vk)
        B_default = barymat(y, xk)
        npt.assert_allclose(np.array(B_default), np.array(B_explicit),
                            rtol=1e-14, atol=1e-15)

    def test_polynomial_exactness(self):
        """B @ f for polynomial of degree < n should be exact."""
        n = 8
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        fk = xk**5 - 3.0 * xk**2 + 1.0
        y = jnp.linspace(-1, 1, 50, dtype=jnp.float64)
        B = barymat(y, xk, vk)
        fx = B @ fk
        exact = y**5 - 3.0 * y**2 + 1.0
        npt.assert_allclose(np.array(fx), np.array(exact), rtol=1e-12, atol=1e-13)

    def test_row_sum_one(self):
        """Each row of B should sum to 1 (partition of unity)."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        y = jnp.linspace(-0.9, 0.9, 30, dtype=jnp.float64)
        B = barymat(y, xk, vk)
        row_sums = jnp.sum(B, axis=1)
        npt.assert_allclose(np.array(row_sums), 1.0, rtol=1e-14, atol=1e-14)

    def test_jit(self):
        """barymat is JIT-compatible (with explicit weights)."""
        n = 10
        xk = chebpts(n, kind=2)
        vk = cheb_bary_weights(n)
        y = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)

        @jax.jit
        def make_B(y, x, w):
            return barymat(y, x, w)

        B_jit = make_B(y, xk, vk)
        B_nojit = barymat(y, xk, vk)
        npt.assert_allclose(np.array(B_jit), np.array(B_nojit), rtol=1e-15)

    @pytest.mark.matlab
    def test_vs_matlab(self, matlab_interpolation):
        """Compare barymat vs MATLAB."""
        for n in [5, 10, 20]:
            xk = chebpts(n, kind=2)
            vk = cheb_bary_weights(n)
            yy = jnp.array(matlab_interpolation[f"barymat_n{n}_m30_yy"],
                           dtype=jnp.float64)
            B = barymat(yy, xk, vk)
            ref = matlab_interpolation[f"barymat_n{n}_m30"]
            npt.assert_allclose(np.array(B), ref, rtol=1e-12, atol=1e-13,
                                err_msg=f"n={n}")

    @pytest.mark.matlab
    def test_default_weights_vs_matlab(self, matlab_interpolation):
        """Compare barymat with default weights vs MATLAB."""
        n = 10
        xk = chebpts(n, kind=2)
        yy = jnp.linspace(-1, 1, 20, dtype=jnp.float64)
        B = barymat(yy, xk)
        ref = matlab_interpolation["barymat_default_n10_m20"]
        npt.assert_allclose(np.array(B), ref, rtol=1e-12, atol=1e-13)
