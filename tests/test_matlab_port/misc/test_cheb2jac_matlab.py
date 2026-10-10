"""Port of MATLAB Chebfun tests/misc/test_cheb2jac.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_cheb2jac.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.transforms import cheb2jac, jac2cheb

TOL = 5e-11


class TestCheb2jac:
    def test_roundtrip(self):
        rng = np.random.default_rng(2)
        c = jnp.asarray(rng.standard_normal(30) / np.arange(1, 31) ** 2)
        back = jac2cheb(cheb2jac(c, 0.2, -0.4), 0.2, -0.4)
        assert float(jnp.max(jnp.abs(back - c))) < TOL

    def test_legendre_special_case(self):
        # P_n^{(0,0)} = Legendre: cheb2jac(c,0,0) == cheb2leg(c)
        from chebfunjax.utils.transforms import cheb2leg
        rng = np.random.default_rng(3)
        c = jnp.asarray(rng.standard_normal(20) / np.arange(1, 21) ** 2)
        a = cheb2jac(c, 0.0, 0.0)
        b = cheb2leg(c)
        assert float(jnp.max(jnp.abs(a - b))) < TOL


class TestNativeParameterContinuity:
    """Active assertions from MATLAB tests/misc/test_cheb2jac.m."""

    def test_native_scalar_n100_parameters(self):
        eps = np.finfo(float).eps
        tol = 100**2 * eps
        c = jnp.asarray(np.random.default_rng(1234).random(100))
        for alpha in (0.5, -0.5):
            actual = cheb2jac(c, alpha, alpha)
            perturbed = cheb2jac(c, alpha + eps, alpha)
            assert float(jnp.max(jnp.abs(actual - perturbed))) < tol

    def test_native_matrix_n513_parameters(self):
        # Native source's final pass(3/4) assignments use C=rand(513), i.e.
        # a 513x513 matrix; the preceding N=100 matrix pass(3/4) is overwritten.
        eps = np.finfo(float).eps
        tol = 100**2 * eps
        c = jnp.asarray(np.random.default_rng(1234).random((513, 513)))
        for alpha in (0.5, -0.5):
            actual = cheb2jac(c, alpha, alpha)
            perturbed = cheb2jac(c, alpha + eps, alpha)
            assert actual.shape == (513, 513)
            assert float(jnp.max(jnp.abs(actual - perturbed))) < tol

    def test_native_matrix_direct_columns(self):
        c = jnp.asarray(np.random.default_rng(19).standard_normal((16, 3)))
        actual = cheb2jac(c, 0.2, -0.4)
        separate = jnp.stack([cheb2jac(c[:, j], 0.2, -0.4) for j in range(3)], axis=1)
        np.testing.assert_allclose(np.asarray(actual), np.asarray(separate), rtol=1e-12, atol=1e-13)
