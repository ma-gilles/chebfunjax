"""Port of MATLAB Chebfun tests/chebop/test_uminusOp.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_uminusOp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-8


class TestChebopUminusOp:
    def test_all_matlab_assertions(self):
        # pass(1): y' - 1 = 0, y(0) = 0 -> y = x, y(1) = 1
        L = Chebop(lambda y: y.diff() - 1.0, domain=(0.0, 1.0))
        L.lbc = 0.0
        y = L.solve(0.0)
        assert abs(float(y(jnp.asarray(1.0))) - 1.0) < TOL

        # pass(2): -y' + 1 = 0 (unary minus inside the op)
        L = Chebop(lambda y: -y.diff() + 1.0, domain=(0.0, 1.0))
        L.lbc = 0.0
        y = L.solve(0.0)
        assert abs(float(y(jnp.asarray(1.0))) - 1.0) < TOL

        # pass(3): -1*y' + 1 = 0
        L = Chebop(lambda y: -1.0 * y.diff() + 1.0, domain=(0.0, 1.0))
        L.lbc = 0.0
        y = L.solve(0.0)
        assert abs(float(y(jnp.asarray(1.0))) - 1.0) < TOL

        # pass(4): 1 - y' = 0
        L = Chebop(lambda y: 1.0 - y.diff(), domain=(0.0, 1.0))
        L.lbc = 0.0
        y = L.solve(0.0)
        assert abs(float(y(jnp.asarray(1.0))) - 1.0) < TOL
