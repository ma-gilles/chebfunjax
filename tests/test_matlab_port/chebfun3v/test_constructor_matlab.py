"""Port of MATLAB Chebfun tests/chebfun3v/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

X0, Y0, Z0 = jnp.asarray(0.3), jnp.asarray(-0.4), jnp.asarray(0.5)


class TestChebfun3vConstructor:
    def test_components(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x * y * z),
                                     lambda x, y, z: x + y,
                                     lambda x, y, z: z * z)
        comps = F.components if hasattr(F, "components") else F
        assert abs(float(comps[0](X0, Y0, Z0))
                   - float(jnp.cos(X0 * Y0 * Z0))) < 1e-10
        assert abs(float(comps[2](X0, Y0, Z0)) - 0.25) < 1e-10
