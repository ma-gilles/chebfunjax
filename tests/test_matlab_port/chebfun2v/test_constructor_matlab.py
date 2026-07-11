"""Port of MATLAB Chebfun tests/chebfun2v/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vConstructor:
    def test_components(self):
        F = Chebfun2v.from_functions(lambda x, y: jnp.cos(x * y),
                                     lambda x, y: x + y)
        comps = F.components if hasattr(F, "components") else F
        assert abs(float(comps[0](X0, Y0))
                   - float(jnp.cos(X0 * Y0))) < 1e-11
        assert abs(float(comps[1](X0, Y0))
                   - float(X0 + Y0)) < 1e-11
