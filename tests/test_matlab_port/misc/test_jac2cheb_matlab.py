"""Port of MATLAB Chebfun tests/misc/test_jac2cheb.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_jac2cheb.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as C
from scipy.special import eval_jacobi

from chebfunjax.utils.transforms import cheb2jac, jac2cheb

TOL = 1e-11


class TestJac2cheb:
    def test_single_jacobi_mode_values(self):
        # jac2cheb(e_k, a, b) must be the Chebyshev coefficients of
        # P_k^{(a,b)}; check values against scipy's eval_jacobi.
        a, b = 0.3, -0.2
        xs = np.linspace(-0.95, 0.95, 40)
        for k in [0, 1, 4, 9]:
            e = jnp.zeros(12, dtype=jnp.float64).at[k].set(1.0)
            cc = np.asarray(jac2cheb(e, a, b))
            got = C.chebval(xs, cc)
            exact = eval_jacobi(k, a, b, xs)
            assert float(np.max(np.abs(got - exact))) < TOL, f"k={k}"


class TestNativeParameterContinuity:
    """Active assertions from MATLAB tests/misc/test_jac2cheb.m."""

    def test_native_scalar_n100_parameters(self):
        eps = np.finfo(float).eps
        tol = 100**2 * eps
        c = jnp.asarray(np.random.default_rng(1234).random(100))
        for alpha in (0.5, -0.5):
            actual = jac2cheb(c, alpha, alpha)
            perturbed = jac2cheb(c, alpha + eps, alpha)
            assert float(jnp.max(jnp.abs(actual - perturbed))) < tol

    def test_native_matrix_n100_parameters(self):
        eps = np.finfo(float).eps
        tol = 100**2 * eps
        c = jnp.asarray(np.random.default_rng(1234).random((100, 100)))
        for alpha in (0.5, -0.5):
            actual = jac2cheb(c, alpha, alpha)
            perturbed = jac2cheb(c, alpha + eps, alpha)
            assert actual.shape == (100, 100)
            assert float(jnp.max(jnp.abs(actual - perturbed))) < tol

    def test_native_matrix_direct_columns(self):
        c = jnp.asarray(np.random.default_rng(20).standard_normal((16, 3)))
        actual = jac2cheb(c, 0.2, -0.4)
        separate = jnp.stack([jac2cheb(c[:, j], 0.2, -0.4) for j in range(3)], axis=1)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(separate), rtol=1e-12, atol=1e-13)

    def test_high_order_matrix_route_roundtrip(self):
        c = jnp.zeros((513, 2), dtype=jnp.float64)
        c = c.at[0, 0].set(1.0).at[2, 1].set(-0.75)
        converted = jac2cheb(c, 0.2, -0.4)
        back = cheb2jac(converted, 0.2, -0.4)
        np.testing.assert_allclose(np.asarray(back), np.asarray(c), rtol=2e-9, atol=2e-11)
