"""Port of MATLAB Chebfun tests/chebop/test_carrier_C1.m (Fable 5).

Carrier equation under the chebcolloc1 discretization (Newton steps
solved via operators/chebop_altdisc).

Provenance
----------
MATLAB source : tests/chebop/test_carrier_C1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10

HIQUALITY = np.array([
    0, -1.487429807540814, -1.785617248281071, 1.572366197526305,
    -1.539652044363185, 1.572366197526230, -1.785617248281089,
    -1.487429807540795, 0])


def _solve_carrier(disc, n):
    dom = (-1.0, 1.0)
    x = cj.chebfun(lambda t: t, domain=dom)
    N = Chebop(lambda xx, u: 0.01 * u.diff(2)
               + 2 * (1 - xx ** 2) * u + u ** 2 - 1, domain=dom)
    N.bc = lambda xx, u: [u(jnp.asarray(-1.0)), u(jnp.asarray(1.0))]
    N.init = 2 * (x ** 2 - 1) * (1 - 2 / (1 + 20 * x ** 2))
    return N.solve(0.0, n=n, discretization=disc)


class TestChebopCarrierC1:
    @pytest.mark.timeout(880)
    def test_all_matlab_assertions(self):
        u = _solve_carrier("chebcolloc1", 256)
        xx = jnp.asarray(np.arange(-1, 1.01, 0.25))
        assert float(np.linalg.norm(np.asarray(u(xx))
                                    - HIQUALITY)) < TOL  # pass(1)
        ends = jnp.asarray([-1.0, 1.0])
        assert float(np.linalg.norm(np.asarray(u(ends)))) < TOL
