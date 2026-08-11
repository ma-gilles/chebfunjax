"""Port of MATLAB Chebfun tests/linop/test_operarith.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_operarith.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.blocks import primitive_operators

jax.config.update("jax_enable_x64", True)

EPS = 2.220446049250313e-16


class TestLinopOperarith:
    def test_all_matlab_assertions(self):
        d = (-1.0, 4.0)
        Z, I, D, C, M = primitive_operators(d)
        f = cj.chebfun(lambda x: jnp.exp(jnp.sin(x) ** 2 + 2), domain=d)

        F = M(f)
        A = -(2 * D ** 2 - F * C + 3 * I)
        Af = A * f

        expected = f * f.cumsum() - 2 * f.diff(2) - 3 * f
        assert float((Af - expected).norm()) < 1e4 * EPS
