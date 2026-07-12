"""Port of MATLAB Chebfun tests/chebfun/test_legcoeffs.m (Fable 5).

FIXED: Chebfun.legcoeffs added in the Fable 5 audit (cheb2leg on the
chebfun coefficients).  Array-valued and piecewise cases skipped
(single-piece only).

Provenance
----------
MATLAB source : tests/chebfun/test_legcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import eval_legendre

import chebfunjax as cj

TOL = 10 * np.finfo(float).eps


class TestChebfunLegcoeffs:
    def test_legendre_polynomials_are_unit_vectors(self):
        # pass(1)
        for n in range(5):
            f = cj.chebfun(
                lambda x, n=n: jnp.asarray(
                    eval_legendre(n, np.asarray(x))))
            c = np.asarray(f.legcoeffs(5))
            e = np.zeros(5)
            e[n] = 1.0
            assert np.max(np.abs(c - e)) < 100 * TOL, n

    def test_linear_combination(self):
        # pass(2)
        v = np.arange(1, 11, dtype=float)

        def f(x):
            xn = np.asarray(x)
            return jnp.asarray(sum(
                v[n] * eval_legendre(n, xn) for n in range(10)))

        c = np.asarray(cj.chebfun(f).legcoeffs(10))
        assert np.max(np.abs(c - v)) < 1000 * TOL

    def test_smooth_reference_values(self):
        # pass(3), first column (sin(x - 0.3))
        f = cj.chebfun(lambda x: jnp.sin(x - 0.3))
        c = np.asarray(f.legcoeffs(5))
        c_exact = np.array([
            -0.2486716793299505096289379,
            0.8631522851187122776801783,
            0.0916630569532407635738723,
            -0.0602302090841307493414730,
            -0.0026889804057628215030252])
        assert np.max(np.abs(c - c_exact)) < 100 * TOL
