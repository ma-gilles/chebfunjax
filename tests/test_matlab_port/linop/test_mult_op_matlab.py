"""Port of MATLAB Chebfun tests/linop/test_mult_op.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_mult_op.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocks import mult

jax.config.update("jax_enable_x64", True)

TOL = 1e-14


def _chebpoly_quasimatrix(orders, dom):
    """Array-valued Chebfun whose columns are ``T_n`` for ``n`` in orders."""
    nmax = max(orders)
    coeffs = jnp.zeros((nmax + 1, len(orders)), dtype=jnp.float64)
    for j, n in enumerate(orders):
        coeffs = coeffs.at[n, j].set(1.0)
    return Chebfun.from_coeffs(coeffs, domain=Domain(dom))


class TestLinopMultOp:
    def test_all_matlab_assertions(self):
        d = (0.0, 2.0)
        x = cj.chebfun(lambda t: t, domain=d)
        f = (2 * x).exp().sin()
        g = x ** 3 - x.cos()

        err = []

        # No breaks.
        F = mult(f)
        err.append(float((F * g - f * g).norm()))

        # Breakpoint.
        f = abs(f)
        F = mult(f)
        err.append(float((F * g - f * g).norm()))

        # Multiplication operator applied to an array-valued CHEBFUN.
        h = x.cos()
        M = mult(h)
        V = _chebpoly_quasimatrix([1, 2, 3, 4], d)
        diff = M * V - h * V
        err.append(float(jnp.max(jnp.abs(jnp.asarray(diff.norm())))))

        assert all(e < TOL for e in err), err
