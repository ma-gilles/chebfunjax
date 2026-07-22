"""Port of MATLAB Chebfun tests/chebfun3v/test_jacobian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_jacobian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 100 * EPS


class TestChebfun3vJacobian:
    def test_definition(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        # For this diagonal field, the Jacobian determinant reduces to
        # F1_x .* F2_y .* F3_z (MATLAB test uses diffx/diffy/diffz).
        Fx = F.diffx()
        Fy = F.diffy()
        Fz = F.diffz()
        jacF = Fx[0] * Fy[1] * Fz[2]
        assert float((jacF - F.jacobian()).norm()) < TOL
