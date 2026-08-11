"""Port of MATLAB Chebfun tests/linop/test_integralops.m (Fable 5).

Ports the chebcolloc2 pass (MATLAB k = 1).  The chebcolloc1 pass is covered
by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_integralops.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import I, fred_op, volt_op

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


class TestLinopIntegralOps:
    def test_all_matlab_assertions(self):
        err = []

        # Fredholm.
        d = (0.0, 1.0)
        x = cj.chebfun(lambda t: t, domain=d)
        F = fred_op(lambda a, b: jnp.sin(2 * jnp.pi * (a - b)), d)
        A = linop(I(d) + F)
        u = x * x.exp()
        f = A * u
        v = A.linsolve([f[0]])
        err.append(float((u - v[0]).norm()))

        # Volterra.
        d = (0.0, math.pi)
        x = cj.chebfun(lambda t: t, domain=d)
        V = volt_op(lambda a, b: a * b, d)
        f = x ** 2 * x.cos() + (1 - x) * x.sin()
        A = linop(I(d) - V)
        u = A.linsolve([f])
        Au = A * u
        err.append(float((u[0] - x.sin()).norm()))
        err.append(float((Au[0] - f).norm()))

        assert all(e < TOL for e in err), err

    @pytest.mark.skip(
        reason="MATLAB's k = 2 pass repeats both solves with the chebcolloc1 "
               "discretization; chebfunjax's BlockLinop only implements "
               "chebcolloc2 rectangular collocation.")
    def test_chebcolloc1(self):
        raise NotImplementedError
