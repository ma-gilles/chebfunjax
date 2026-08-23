"""Port of MATLAB Chebfun tests/chebop/test_basic_arithmetic.m (Fable 5).

MATLAB's ``eye(A)`` maps to ``A.eye()``.

Provenance
----------
MATLAB source : tests/chebop/test_basic_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _n(f):
    xs = jnp.linspace(0.0 + 1e-9, 3.0 - 1e-9, 33)
    return float(jnp.max(jnp.abs(jnp.asarray(f(xs)))))


class TestChebopBasicArithmetic:
    def test_all_matlab_assertions(self):
        dom = (0.0, 3.0)
        u = cj.chebfun(jnp.sin, domain=dom)
        A = Chebop(lambda u: u, domain=dom)
        B = Chebop(lambda u: u, domain=dom)
        C = A + B
        assert _n(A(u) + B(u) - C(u)) == 0.0        # pass(1)
        C = A - B
        assert _n(C(u)) == 0.0                      # pass(2)
        C = -B
        assert _n(B(u) + C(u)) == 0.0               # pass(3)
        C = 2 * A
        assert _n(2 * A(u) - C(u)) == 0.0           # pass(4)/(5)
        # MATLAB A/2 (mrdivide); chebfunjax reserves / for solve, so
        # the scalar division is spelled as multiplication by 0.5.
        C = A * 0.5
        assert _n(A(u) / 2 - C(u)) == 0.0           # pass(6)/(7)
        I = A.eye()
        assert _n(u - I(u)) == 0.0                  # pass(8)
        assert _n((A - I) * u - (A(u) - u)) == 0.0  # pass(9)
