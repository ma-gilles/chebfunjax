"""Port of MATLAB Chebfun tests/chebfun3v/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

X0, Y0, Z0 = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(0.5)


class TestChebfun3vCross:
    def test_xhat_cross_yhat(self):
        E1 = Chebfun3v.from_functions(lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x,
                                      lambda x, y, z: 0 * x)
        E2 = Chebfun3v.from_functions(lambda x, y, z: 0 * x,
                                      lambda x, y, z: 1.0 + 0 * x,
                                      lambda x, y, z: 0 * x)
        C = E1.cross(E2)
        comps = C.components if hasattr(C, "components") else C
        assert abs(float(comps[0](X0, Y0, Z0))) < 1e-10
        assert abs(float(comps[1](X0, Y0, Z0))) < 1e-10
        assert abs(float(comps[2](X0, Y0, Z0)) - 1.0) < 1e-10
