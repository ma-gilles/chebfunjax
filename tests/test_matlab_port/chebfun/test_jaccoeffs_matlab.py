"""Port of MATLAB Chebfun tests/chebfun/test_jaccoeffs.m (Fable 5).

FIXED: Chebfun.jaccoeffs added in the Fable 5 audit (cheb2jac).

Provenance
----------
MATLAB source : tests/chebfun/test_jaccoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import eval_jacobi

import chebfunjax as cj

TOL = 1e-12


class TestChebfunJaccoeffs:
    def test_jacobi_polynomials_are_unit_vectors(self):
        a, b = 0.4, -0.3
        for n in range(5):
            f = cj.chebfun(
                lambda x, n=n: jnp.asarray(
                    eval_jacobi(n, a, b, np.asarray(x))))
            c = np.asarray(f.jaccoeffs(5, a, b))
            e = np.zeros(5)
            e[n] = 1.0
            assert np.max(np.abs(c - e)) < TOL, n

    def test_alpha_beta_zero_is_legendre(self):
        f = cj.chebfun(lambda x: jnp.sin(x - 0.3))
        cj_ = np.asarray(f.jaccoeffs(5, 0.0, 0.0))
        cl = np.asarray(f.legcoeffs(5))
        assert np.max(np.abs(cj_ - cl)) < TOL
