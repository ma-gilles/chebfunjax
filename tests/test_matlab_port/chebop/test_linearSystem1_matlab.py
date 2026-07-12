"""Port of MATLAB Chebfun tests/chebop/test_linearSystem1.m (Fable 5).

FIXED: linear systems of coupled ODEs added in the Fable 5 audit
(block collocation via basis probing; Chebop ops with signature
(x, u, v, ...) dispatch to the system solver).  The chebcolloc1 /
ultraS discretization variants are covered by the single chebfunjax
collocation; the piecewise-domain case remains a documented skip.

Provenance
----------
MATLAB source : tests/chebop/test_linearSystem1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


class TestChebopLinearSystem1:
    def test_2x2_sin_cos_system(self):
        d = (-np.pi, np.pi)
        A = Chebop(lambda x, u, v: [u - v.diff(), u.diff() + v], d)
        A.lbc = lambda u, v: u + 1
        A.rbc = lambda u, v: v
        sol = A.solve(0)
        u1, u2 = sol[0], sol[1]
        xs = jnp.asarray(np.linspace(-0.99 * np.pi, 0.99 * np.pi, 60))
        # pass(4)-(5): u = cos, v = sin
        assert float(jnp.max(jnp.abs(u1(xs) - jnp.cos(xs)))) \
            < 100 * TOL
        assert float(jnp.max(jnp.abs(u2(xs) - jnp.sin(xs)))) \
            < 100 * TOL
        # pass(6): boundary residuals
        assert abs(float(u1(jnp.asarray(d[0]))) + 1) < 10 * TOL
        assert abs(float(u2(jnp.asarray(d[1])))) < 10 * TOL
