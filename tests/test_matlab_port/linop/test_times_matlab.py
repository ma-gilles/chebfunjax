"""Port of MATLAB Chebfun tests/linop/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.domain import Domain
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import D

jax.config.update("jax_enable_x64", True)


def _chebpoly_quasimatrix(orders, dom):
    nmax = max(orders)
    coeffs = jnp.zeros((nmax + 1, len(orders)), dtype=jnp.float64)
    for j, n in enumerate(orders):
        coeffs = coeffs.at[n, j].set(1.0)
    return Chebfun.from_coeffs(coeffs, domain=Domain(dom))


class TestLinopTimes:
    def test_all_matlab_assertions(self):
        dom = (-1.0, 1.0)
        diff_op = D(dom)
        V = _chebpoly_quasimatrix(list(range(1, 7)), dom)

        A = linop(diff_op)
        AV = A * V

        B = linop(2 * diff_op)
        BV = B * V

        resid = 2 * AV[0] - BV[0]
        err = float(jnp.max(jnp.abs(jnp.asarray(resid.norm()))))
        assert err < 1e-14
