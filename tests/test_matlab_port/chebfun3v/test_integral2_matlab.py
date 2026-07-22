"""Port of MATLAB Chebfun tests/chebfun3v/test_integral2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_integral2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e4 * EPS


class TestChebfun3vIntegral2:
    def test_flux_integral(self):
        # Surface: unit disk in the xy-plane, parametrised (r, phi).
        S = Chebfun2v.from_functions(
            lambda r, phi: r * jnp.cos(phi),
            lambda r, phi: r * jnp.sin(phi),
            lambda r, phi: 0 * r,
            domain=(0, 1, 0, 2 * np.pi))
        # Constant vector field [0; 0; 1]; flux through the disk is pi.
        F = Chebfun3v.from_functions(lambda x, y, z: 0 * x,
                                     lambda x, y, z: 0 * y,
                                     lambda x, y, z: 0 * x + 1)
        I = F.integral2(S)
        assert abs(float(I) - np.pi) < TOL
