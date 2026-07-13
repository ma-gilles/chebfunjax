"""Port of MATLAB Chebfun tests/chebfun/test_trigremez.m (Fable 5).

FIXED: trigremez (periodic Remez, Javed & Trefethen; polynomial
case) added in the Fable 5 audit (Big-Three trig-rational
directive).  The rational case remains a documented gap.

Provenance
----------
MATLAB source : tests/chebfun/test_trigremez.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

TOL = 1e-10
TT = jnp.asarray(np.linspace(0.01, 1.99, 60))


class TestChebfunTrigremez:
    def test_best_approx_and_equioscillation(self):
        f = cj.chebfun(lambda x: jnp.cos(4 * np.pi * x) + 1,
                       domain=(0.0, 2.0), trig=True)
        # pass(2)-(3): degree 1 -> the constant 1, equioscillating
        p, errmax, status = cj.trigremez(f, 1)
        assert float(jnp.max(jnp.abs(p(TT) - 1.0))) < 100 * TOL
        xk = jnp.asarray(status["xk"])
        err_ref = np.abs(np.asarray(f(xk)) - np.asarray(p(xk)))
        assert float(np.max(np.abs(err_ref - errmax))) < 100 * TOL

    def test_degree_4_reproduces(self):
        # pass(4)-(5) analogue: degree 4 reproduces the function
        f = cj.chebfun(lambda x: jnp.cos(4 * np.pi * x) + 1,
                       domain=(0.0, 2.0), trig=True)
        p, errmax, _ = cj.trigremez(f, 4)
        assert float(jnp.max(jnp.abs(p(TT) - f(TT)))) < 100 * TOL
        assert errmax < 100 * TOL
