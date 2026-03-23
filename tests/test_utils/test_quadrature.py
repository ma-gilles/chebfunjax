"""Tests for chebfunjax.utils.quadrature — Chebyshev and Legendre points/weights."""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

from chebfunjax.utils.quadrature import chebpts, chebpts_ab, chebweights


class TestChebpts:
    """Tests for Chebyshev points."""

    def test_chebpts2_n5(self):
        """5 Chebyshev points of the 2nd kind."""
        x = chebpts(5, kind=2)
        expected = np.cos(np.arange(4, -1, -1) * np.pi / 4)
        npt.assert_allclose(np.array(x), expected, rtol=1e-14)

    def test_chebpts2_endpoints(self):
        """2nd-kind points include ±1."""
        for n in [3, 5, 10, 50]:
            x = chebpts(n, kind=2)
            npt.assert_allclose(float(x[0]), -1.0, atol=1e-15)
            npt.assert_allclose(float(x[-1]), 1.0, atol=1e-15)

    def test_chebpts1_no_endpoints(self):
        """1st-kind points do not include ±1."""
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


class TestJITCompatibility:
    """Verify functions work under jax.jit."""

    def test_chebpts_jit(self):
        """chebpts is not fully JIT-able (n and kind control branching),
        but the core array ops work under JIT when called with static args."""
        import functools
        jitted = jax.jit(functools.partial(chebpts, 10, kind=2))
        npt.assert_allclose(np.array(jitted()), np.array(chebpts(10, kind=2)), rtol=1e-15)

    def test_chebweights_jit(self):
        @jax.jit
        def f():
            return chebweights(10, kind=2)
        npt.assert_allclose(np.array(f()), np.array(chebweights(10, kind=2)), rtol=1e-15)
