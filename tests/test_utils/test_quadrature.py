"""Tests for chebfunjax.utils.quadrature — Chebyshev and Legendre points/weights."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.quadrature import (
    chebpts,
    chebpts_ab,
    chebweights,
    hermpts,
    jacpts,
    lagpts,
    legpts,
    lobpts,
    radaupts,
    trigpts,
    ultrapts,
)


class TestChebpts:
    """Tests for Chebyshev points."""

    def test_chebpts2_n5(self):
        """5 Chebyshev points of the 2nd kind."""
        x = chebpts(5, kind=2)
        expected = np.cos(np.arange(4, -1, -1) * np.pi / 4)
        npt.assert_allclose(np.array(x), expected, rtol=1e-14)

    def test_chebpts2_endpoints(self):
        """2nd-kind points include +/-1."""
        for n in [3, 5, 10, 50]:
            x = chebpts(n, kind=2)
            npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)
            npt.assert_allclose(float(x[-1]), 1.0, atol=1e-15)

    def test_chebpts1_no_endpoints(self):
        """1st-kind points do not include +/-1."""
        for n in [3, 5, 10]:
            x = chebpts(n, kind=1)
            assert float(x[0]) > -1.0
            assert float(x[-1]) < 1.0

    def test_chebpts1_n1(self):
        x = chebpts(1, kind=1)
        npt.assert_allclose(np.array(x), [0.0], atol=1e-15)

    def test_chebpts_n0(self):
        x = chebpts(0)
        assert len(x) == 0

    def test_chebpts_symmetry(self):
        """Points should be symmetric about 0."""
        for n in [5, 10, 17, 32]:
            for kind in [1, 2]:
                x = chebpts(n, kind=kind)
                npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-14)

    @pytest.mark.matlab
    def test_chebpts2_vs_matlab(self, matlab_quadrature):
        """Compare against MATLAB chebpts output."""
        for n in [5, 10, 17, 32, 64, 128]:
            x = chebpts(n, kind=2)
            ref = matlab_quadrature[f"chebpts2_n{n}"]
            npt.assert_allclose(np.array(x), ref, rtol=1e-14, atol=1e-15)

    @pytest.mark.matlab
    def test_chebpts1_vs_matlab(self, matlab_quadrature):
        for n in [5, 10, 17, 32, 64, 128]:
            x = chebpts(n, kind=1)
            ref = matlab_quadrature[f"chebpts1_n{n}"]
            npt.assert_allclose(np.array(x), ref, rtol=1e-14, atol=1e-15)


class TestChebptsAB:
    def test_scaled_interval(self):
        x = chebpts_ab(5, 0.0, 2.0)
        npt.assert_allclose(float(x[0]), 0.0, atol=1e-14)
        npt.assert_allclose(float(x[-1]), 2.0, atol=1e-14)


class TestChebweights:
    def test_cc_integrates_1(self):
        """Clenshaw-Curtis weights should integrate f(x)=1 exactly."""
        for n in [2, 5, 10, 50]:
            w = chebweights(n, kind=2)
            npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-14)

    def test_cc_integrates_x2(self):
        """Clenshaw-Curtis should integrate x^2 on [-1,1] exactly = 2/3."""
        for n in [3, 5, 10]:
            x = chebpts(n, kind=2)
            w = chebweights(n, kind=2)
            integral = float(jnp.dot(w, x**2))
            npt.assert_allclose(integral, 2.0 / 3.0, rtol=1e-13)

    def test_cc_integrates_polynomial(self):
        """CC with n points integrates degree 2n-3 polynomials exactly."""
        n = 10
        x = chebpts(n, kind=2)
        w = chebweights(n, kind=2)
        # x^(2n-3) = x^17 for n=10
        deg = 2 * n - 3
        integral = float(jnp.dot(w, x**deg))
        # Exact integral of x^17 on [-1,1] = 0 (odd function)
        npt.assert_allclose(integral, 0.0, atol=1e-12)

    def test_gc_integrates_1(self):
        """Gauss-Chebyshev weights: sum should = pi (integral of 1/sqrt(1-x^2))."""
        for n in [5, 10, 50]:
            w = chebweights(n, kind=1)
            npt.assert_allclose(float(jnp.sum(w)), jnp.pi, rtol=1e-14)


# ===========================================================================
# Gauss-Legendre
# ===========================================================================


class TestLegpts:
    """Tests for Gauss-Legendre quadrature.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = legpts(0)
        assert len(x) == 0
        assert len(w) == 0

    def test_n1(self):
        x, w = legpts(1)
        npt.assert_allclose(np.array(x), [0.0], atol=1e-15)
        npt.assert_allclose(np.array(w), [2.0], atol=1e-15)

    def test_n2(self):
        x, w = legpts(2)
        npt.assert_allclose(np.array(x), [-1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
                            rtol=1e-14)
        npt.assert_allclose(np.array(w), [1.0, 1.0], rtol=1e-14)

    def test_symmetry(self):
        """Legendre points are symmetric about 0."""
        for n in [5, 10, 17, 32]:
            x, _ = legpts(n)
            npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-13)

    def test_weights_sum(self):
        """Weights should sum to 2 (integral of 1 on [-1,1])."""
        for n in [3, 5, 10, 32, 64]:
            _, w = legpts(n)
            npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-13)

    def test_exactness_degree_2n_minus_1(self):
        """Gauss-Legendre with n points integrates polynomials of degree <= 2n-1 exactly."""
        for n in [3, 5, 8, 10]:
            x, w = legpts(n)
            deg = 2 * n - 1
            # Integral of x^deg on [-1,1]
            if deg % 2 == 0:
                exact = 2.0 / (deg + 1)
            else:
                exact = 0.0
            integral = float(jnp.dot(w, x**deg))
            npt.assert_allclose(integral, exact, atol=1e-12,
                                err_msg=f"n={n}, deg={deg}")

    def test_exactness_even_polynomial(self):
        """Integrate x^4 on [-1,1] = 2/5 (exact for n >= 3)."""
        for n in [3, 5, 10]:
            x, w = legpts(n)
            npt.assert_allclose(float(jnp.dot(w, x**4)), 2.0 / 5.0, rtol=1e-13)

    def test_interval_scaling(self):
        """Nodes and weights rescale correctly to [0, 2]."""
        n = 10
        x, w = legpts(n, interval=(0.0, 2.0))
        npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-13)
        # Integrate x^2 on [0,2] = 8/3
        npt.assert_allclose(float(jnp.dot(w, x**2)), 8.0 / 3.0, rtol=1e-12)

    @pytest.mark.matlab
    def test_legpts_x_vs_matlab(self, matlab_quadrature):
        """Compare nodes against MATLAB legpts."""
        for n in [5, 10, 17, 32, 64, 128]:
            x, _ = legpts(n)
            ref = matlab_quadrature[f"legpts_x_n{n}"]
            npt.assert_allclose(np.array(x), ref, rtol=1e-12, atol=1e-14,
                                err_msg=f"legpts nodes n={n}")

    @pytest.mark.matlab
    def test_legpts_w_vs_matlab(self, matlab_quadrature):
        """Compare weights against MATLAB legpts."""
        for n in [5, 10, 17, 32, 64, 128]:
            _, w = legpts(n)
            ref = matlab_quadrature[f"legpts_w_n{n}"]
            npt.assert_allclose(np.array(w), ref, rtol=1e-12, atol=1e-14,
                                err_msg=f"legpts weights n={n}")

    def test_jit(self):
        """JIT with static n."""
        jitted = jax.jit(functools.partial(legpts, 10))
        x_jit, w_jit = jitted()
        x_ref, w_ref = legpts(10)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Gauss-Jacobi
# ===========================================================================


class TestJacpts:
    """Tests for Gauss-Jacobi quadrature.

    JAX contract: jit=yes (n, a, b must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = jacpts(0, 0.5, 0.5)
        assert len(x) == 0
        assert len(w) == 0

    def test_n1(self):
        x, w = jacpts(1, 0.5, 1.5)
        assert x.shape == (1,)
        assert w.shape == (1,)

    def test_symmetry_alpha_eq_beta(self):
        """When alpha == beta, nodes are symmetric about 0."""
        for n in [5, 10]:
            for ab in [0.5, 1.0, 2.0]:
                x, _ = jacpts(n, ab, ab)
                npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-12,
                                    err_msg=f"n={n}, a=b={ab}")

    def test_legendre_special_case(self):
        """jacpts(n, 0, 0) should match legpts(n)."""
        for n in [5, 10, 17]:
            xj, wj = jacpts(n, 0.0, 0.0)
            xl, wl = legpts(n)
            npt.assert_allclose(np.array(xj), np.array(xl), rtol=1e-13)
            npt.assert_allclose(np.array(wj), np.array(wl), rtol=1e-13)

    def test_weight_sum(self):
        """Weights should sum to integral of (1-x)^a * (1+x)^b on [-1,1]."""
        import jax.scipy.special as jsp
        for n in [5, 10]:
            for a, b in [(0.5, 0.5), (1.0, 1.0), (0.5, 1.5)]:
                _, w = jacpts(n, a, b)
                # Integral = 2^(a+b+1) * Beta(a+1, b+1)
                exact = float(2.0 ** (a + b + 1.0) * jnp.exp(
                    jsp.gammaln(a + 1.0) + jsp.gammaln(b + 1.0)
                    - jsp.gammaln(a + b + 2.0)))
                npt.assert_allclose(float(jnp.sum(w)), exact, rtol=1e-11,
                                    err_msg=f"n={n}, a={a}, b={b}")

    def test_exactness(self):
        """Gauss-Jacobi with n points integrates x^k * w(x) for k <= 2n-1."""
        a, b = 0.5, 1.5
        n = 5
        x, w = jacpts(n, a, b)
        # Integrate x^2 * (1-x)^a * (1+x)^b on [-1,1]
        computed = float(jnp.dot(w, x**2))
        # For simple check, just verify it's close to a known integral
        # Use higher-n for reference
        x_ref, w_ref = jacpts(20, a, b)
        ref = float(jnp.dot(w_ref, x_ref**2))
        npt.assert_allclose(computed, ref, rtol=1e-11)

    @pytest.mark.matlab
    def test_jacpts_vs_matlab(self, matlab_quadrature):
        """Compare against MATLAB jacpts output."""
        for n in [5, 10, 17, 32, 64]:
            for a, b in [(0.5, 0.5), (1.0, 1.0), (0.5, 1.5), (2.0, 0.0)]:
                tag = f"jacpts_a{a:.1f}_b{b:.1f}_n{n}".replace(".", "p")
                x, w = jacpts(n, a, b)
                ref_x = matlab_quadrature[f"{tag}_x"]
                ref_w = matlab_quadrature[f"{tag}_w"]
                npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                    err_msg=f"jacpts x: a={a}, b={b}, n={n}")
                npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                    err_msg=f"jacpts w: a={a}, b={b}, n={n}")

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            jacpts(5, -1.5, 0.0)

    def test_jit(self):
        jitted = jax.jit(functools.partial(jacpts, 10, 0.5, 1.5))
        x_jit, w_jit = jitted()
        x_ref, w_ref = jacpts(10, 0.5, 1.5)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-13)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-13)


# ===========================================================================
# Gauss-Hermite
# ===========================================================================


class TestHermpts:
    """Tests for Gauss-Hermite quadrature.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = hermpts(0)
        assert len(x) == 0
        assert len(w) == 0

    def test_n1(self):
        x, w = hermpts(1)
        npt.assert_allclose(np.array(x), [0.0], atol=1e-15)
        npt.assert_allclose(np.array(w), [np.sqrt(np.pi)], rtol=1e-14)

    def test_symmetry(self):
        """Hermite points are symmetric about 0."""
        for n in [5, 10, 17]:
            x, _ = hermpts(n)
            npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-13)

    def test_weights_sum_phys(self):
        """Weights should sum to sqrt(pi) for physicist's Hermite."""
        for n in [3, 5, 10, 32]:
            _, w = hermpts(n, kind="phys")
            npt.assert_allclose(float(jnp.sum(w)), np.sqrt(np.pi), rtol=1e-13)

    def test_weights_sum_prob(self):
        """Weights should sum to sqrt(2*pi) for probabilist's Hermite."""
        for n in [3, 5, 10, 32]:
            _, w = hermpts(n, kind="prob")
            npt.assert_allclose(float(jnp.sum(w)), np.sqrt(2.0 * np.pi), rtol=1e-13)

    def test_exactness(self):
        """n-point rule integrates x^k * exp(-x^2) for k <= 2n-1."""
        n = 5
        x, w = hermpts(n)
        # Integral of x^2 * exp(-x^2) on (-inf, inf) = sqrt(pi)/2
        npt.assert_allclose(float(jnp.dot(w, x**2)),
                            np.sqrt(np.pi) / 2.0, rtol=1e-13)
        # Integral of x^4 * exp(-x^2) on (-inf, inf) = 3*sqrt(pi)/4
        npt.assert_allclose(float(jnp.dot(w, x**4)),
                            3.0 * np.sqrt(np.pi) / 4.0, rtol=1e-12)

    @pytest.mark.matlab
    def test_hermpts_vs_matlab(self, matlab_quadrature):
        for n in [5, 10, 17, 32, 64]:
            x, w = hermpts(n)
            ref_x = matlab_quadrature[f"hermpts_x_n{n}"]
            ref_w = matlab_quadrature[f"hermpts_w_n{n}"]
            npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                err_msg=f"hermpts x, n={n}")
            npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                err_msg=f"hermpts w, n={n}")

    def test_invalid_kind(self):
        with pytest.raises(ValueError):
            hermpts(5, kind="invalid")

    def test_jit(self):
        jitted = jax.jit(functools.partial(hermpts, 10))
        x_jit, w_jit = jitted()
        x_ref, w_ref = hermpts(10)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Gauss-Laguerre
# ===========================================================================


class TestLagpts:
    """Tests for Gauss-Laguerre quadrature.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = lagpts(0)
        assert len(x) == 0
        assert len(w) == 0

    def test_weights_sum(self):
        """Weights should sum to Gamma(alpha+1) = 1 for alpha=0."""
        for n in [3, 5, 10, 32]:
            _, w = lagpts(n)
            npt.assert_allclose(float(jnp.sum(w)), 1.0, rtol=1e-12)

    def test_nodes_positive(self):
        """All Laguerre nodes should be positive."""
        for n in [5, 10, 32]:
            x, _ = lagpts(n)
            assert float(jnp.min(x)) > 0.0

    def test_exactness(self):
        """n-point rule integrates x^k * exp(-x) for k <= 2n-1."""
        n = 5
        x, w = lagpts(n)
        # Integral of x * exp(-x) on [0, inf) = 1
        npt.assert_allclose(float(jnp.dot(w, x)), 1.0, rtol=1e-12)
        # Integral of x^2 * exp(-x) on [0, inf) = 2
        npt.assert_allclose(float(jnp.dot(w, x**2)), 2.0, rtol=1e-12)

    @pytest.mark.matlab
    def test_lagpts_vs_matlab(self, matlab_quadrature):
        for n in [5, 10, 17, 32, 64]:
            x, w = lagpts(n)
            ref_x = matlab_quadrature[f"lagpts_x_n{n}"]
            ref_w = matlab_quadrature[f"lagpts_w_n{n}"]
            npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                err_msg=f"lagpts x, n={n}")
            npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                err_msg=f"lagpts w, n={n}")

    def test_jit(self):
        jitted = jax.jit(functools.partial(lagpts, 10))
        x_jit, w_jit = jitted()
        x_ref, w_ref = lagpts(10)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Ultraspherical (Gegenbauer)
# ===========================================================================


class TestUltrapts:
    """Tests for Gauss-Gegenbauer quadrature.

    JAX contract: jit=yes (n, lam must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = ultrapts(0, 0.75)
        assert len(x) == 0
        assert len(w) == 0

    def test_n1(self):
        x, w = ultrapts(1, 0.75)
        assert x.shape == (1,)
        npt.assert_allclose(float(x[0]), 0.0, atol=1e-15)

    def test_symmetry(self):
        """Ultraspherical points are symmetric about 0."""
        for n in [5, 10]:
            for lam in [0.75, 1.5, 2.5]:
                x, _ = ultrapts(n, lam)
                npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-12)

    def test_legendre_special_case(self):
        """ultrapts(n, 0.5) should match legpts(n)."""
        for n in [5, 10, 17]:
            xu, wu = ultrapts(n, 0.5)
            xl, wl = legpts(n)
            npt.assert_allclose(np.array(xu), np.array(xl), rtol=1e-13)
            npt.assert_allclose(np.array(wu), np.array(wl), rtol=1e-13)

    def test_weight_sum(self):
        """Weights sum to integral of (1-x^2)^(lam-1/2) on [-1,1]."""
        import jax.scipy.special as jsp
        for n in [10, 20]:
            for lam in [0.75, 1.5, 2.5]:
                _, w = ultrapts(n, lam)
                # Exact: sqrt(pi) * Gamma(lam+1/2) / Gamma(lam+1)
                exact = float(jnp.sqrt(jnp.pi) * jnp.exp(
                    jsp.gammaln(lam + 0.5) - jsp.gammaln(lam + 1.0)))
                npt.assert_allclose(float(jnp.sum(w)), exact, rtol=1e-11,
                                    err_msg=f"n={n}, lam={lam}")

    @pytest.mark.matlab
    def test_ultrapts_vs_matlab(self, matlab_quadrature):
        for n in [5, 10, 17, 32, 64]:
            for lam in [0.75, 1.5, 2.5]:
                tag = f"ultrapts_lam{lam:.2f}_n{n}".replace(".", "p")
                x, w = ultrapts(n, lam)
                ref_x = matlab_quadrature[f"{tag}_x"]
                ref_w = matlab_quadrature[f"{tag}_w"]
                npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                    err_msg=f"ultrapts x: lam={lam}, n={n}")
                npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                    err_msg=f"ultrapts w: lam={lam}, n={n}")

    def test_invalid_lambda(self):
        with pytest.raises(ValueError):
            ultrapts(5, -1.0)

    def test_jit(self):
        jitted = jax.jit(functools.partial(ultrapts, 10, 1.5))
        x_jit, w_jit = jitted()
        x_ref, w_ref = ultrapts(10, 1.5)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Gauss-Radau
# ===========================================================================


class TestRadaupts:
    """Tests for Gauss-Radau quadrature.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n1(self):
        x, w = radaupts(1)
        npt.assert_allclose(np.array(x), [-1.0], atol=1e-15)
        npt.assert_allclose(np.array(w), [2.0], rtol=1e-14)

    def test_left_endpoint(self):
        """Radau always includes x = -1."""
        for n in [3, 5, 10]:
            x, _ = radaupts(n)
            npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)

    def test_no_right_endpoint(self):
        """Radau does not include x = +1."""
        for n in [3, 5, 10]:
            x, _ = radaupts(n)
            assert float(x[-1]) < 1.0

    def test_weight_sum(self):
        """Weights should sum to 2 (Legendre case)."""
        for n in [3, 5, 10, 17]:
            _, w = radaupts(n)
            npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-12)

    @pytest.mark.matlab
    def test_radaupts_vs_matlab(self, matlab_quadrature):
        for n in [3, 5, 10, 17, 32]:
            x, w = radaupts(n)
            ref_x = matlab_quadrature[f"radaupts_x_n{n}"]
            ref_w = matlab_quadrature[f"radaupts_w_n{n}"]
            npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                err_msg=f"radaupts x, n={n}")
            npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                err_msg=f"radaupts w, n={n}")

    def test_jit(self):
        jitted = jax.jit(functools.partial(radaupts, 10))
        x_jit, w_jit = jitted()
        x_ref, w_ref = radaupts(10)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Gauss-Lobatto
# ===========================================================================


class TestLobpts:
    """Tests for Gauss-Lobatto quadrature.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n2(self):
        x, w = lobpts(2)
        npt.assert_allclose(np.array(x), [-1.0, 1.0], atol=1e-15)
        npt.assert_allclose(np.array(w), [1.0, 1.0], rtol=1e-14)

    def test_endpoints(self):
        """Lobatto always includes both +/-1."""
        for n in [3, 5, 10]:
            x, _ = lobpts(n)
            npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)
            npt.assert_allclose(float(x[-1]), 1.0, atol=1e-15)

    def test_weight_sum(self):
        """Weights should sum to 2 (Legendre case)."""
        for n in [3, 5, 10, 17]:
            _, w = lobpts(n)
            npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-12)

    def test_symmetry(self):
        """Lobatto points are symmetric about 0 (Legendre case)."""
        for n in [5, 10]:
            x, _ = lobpts(n)
            npt.assert_allclose(np.array(x + x[::-1]), 0.0, atol=1e-13)

    def test_n_less_than_2(self):
        with pytest.raises(ValueError):
            lobpts(1)

    @pytest.mark.matlab
    def test_lobpts_vs_matlab(self, matlab_quadrature):
        for n in [3, 5, 10, 17, 32]:
            x, w = lobpts(n)
            ref_x = matlab_quadrature[f"lobpts_x_n{n}"]
            ref_w = matlab_quadrature[f"lobpts_w_n{n}"]
            npt.assert_allclose(np.array(x), ref_x, rtol=1e-11, atol=1e-13,
                                err_msg=f"lobpts x, n={n}")
            npt.assert_allclose(np.array(w), ref_w, rtol=1e-11, atol=1e-13,
                                err_msg=f"lobpts w, n={n}")

    def test_jit(self):
        jitted = jax.jit(functools.partial(lobpts, 10))
        x_jit, w_jit = jitted()
        x_ref, w_ref = lobpts(10)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-14)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-14)


# ===========================================================================
# Trigonometric (equispaced)
# ===========================================================================


class TestTrigpts:
    """Tests for equispaced trigonometric points.

    JAX contract: jit=yes (n must be static), vmap=no, grad=no
    """

    def test_n0(self):
        x, w = trigpts(0)
        assert len(x) == 0
        assert len(w) == 0

    def test_n4(self):
        x, w = trigpts(4)
        expected = np.array([-1.0, -0.5, 0.0, 0.5])
        npt.assert_allclose(np.array(x), expected, rtol=1e-14)
        npt.assert_allclose(np.array(w), np.full(4, 0.5), rtol=1e-14)

    def test_weight_sum(self):
        """Trapezoidal weights should sum to 2 on [-1, 1)."""
        for n in [4, 8, 16, 32]:
            _, w = trigpts(n)
            npt.assert_allclose(float(jnp.sum(w)), 2.0, rtol=1e-14)

    def test_interval_scaling(self):
        """Points and weights scale correctly to [0, 2*pi]."""
        n = 8
        x, w = trigpts(n, interval=(0.0, 2.0 * np.pi))
        npt.assert_allclose(float(jnp.sum(w)), 2.0 * np.pi, rtol=1e-13)

    def test_no_right_endpoint(self):
        """Equispaced points on [-1, 1) should not include 1."""
        for n in [4, 8, 16]:
            x, _ = trigpts(n)
            assert float(jnp.max(x)) < 1.0

    @pytest.mark.matlab
    def test_trigpts_vs_matlab(self, matlab_quadrature):
        for n in [4, 8, 16, 32]:
            x, w = trigpts(n)
            ref_x = matlab_quadrature[f"trigpts_x_n{n}"]
            ref_w = matlab_quadrature[f"trigpts_w_n{n}"]
            npt.assert_allclose(np.array(x), ref_x, rtol=1e-14, atol=1e-15,
                                err_msg=f"trigpts x, n={n}")
            npt.assert_allclose(np.array(w), ref_w, rtol=1e-14, atol=1e-15,
                                err_msg=f"trigpts w, n={n}")

    def test_jit(self):
        jitted = jax.jit(functools.partial(trigpts, 8))
        x_jit, w_jit = jitted()
        x_ref, w_ref = trigpts(8)
        npt.assert_allclose(np.array(x_jit), np.array(x_ref), rtol=1e-15)
        npt.assert_allclose(np.array(w_jit), np.array(w_ref), rtol=1e-15)


# ===========================================================================
# JIT compatibility (original tests preserved)
# ===========================================================================


class TestJITCompatibility:
    """Verify functions work under jax.jit."""

    def test_chebpts_jit(self):
        """chebpts is not fully JIT-able (n and kind control branching),
        but the core array ops work under JIT when called with static args."""
        jitted = jax.jit(functools.partial(chebpts, 10, kind=2))
        npt.assert_allclose(np.array(jitted()), np.array(chebpts(10, kind=2)), rtol=1e-15)

    def test_chebweights_jit(self):
        @jax.jit
        def f():
            return chebweights(10, kind=2)
        npt.assert_allclose(np.array(f()), np.array(chebweights(10, kind=2)), rtol=1e-15)
