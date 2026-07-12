"""Port of MATLAB Chebfun tests/chebop/test_system3.m (Fable 5).

FIXED: nonlinear systems of coupled ODEs added in the Fable 5 audit
(Newton with finite-difference block Jacobian on top of the
block-collocation discretization; list-valued lbc/rbc supported).

Provenance
----------
MATLAB source : tests/chebop/test_system3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.operators.chebop import Chebop


class TestChebopSystem3:
    def test_nonlinear_coupled_bvp(self):
        N = Chebop(
            lambda x, u, v: [u.diff(2) + v.sin(),
                             u.cos() + v.diff(2)],
            (-1.0, 1.0))
        N.lbc = lambda u, v: [u - 2, v - 1]
        N.rbc = lambda u, v: [u - 2, v + 1]
        sol = N.solve([0, 0])
        v = sol[1]
        assert abs(float(v(jnp.asarray(0.2)))
                   - (-0.371250985730553)) < 1e-8
