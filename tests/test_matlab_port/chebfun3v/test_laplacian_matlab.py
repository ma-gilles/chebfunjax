"""Port of MATLAB Chebfun tests/chebfun3v/test_laplacian.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_laplacian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e2 * EPS


def _lap_component(c):
    return c.diff(1, 2) + c.diff(2, 2) + c.diff(3, 2)


class TestChebfun3vLaplacian:
    def test_definition_diagonal(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        lapF = F.laplacian()
        for j in range(3):
            assert float((_lap_component(F[j]) - lapF[j]).norm()) < TOL

    def test_definition_mixed(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: x * y + y ** 2)
        lapF = F.laplacian()
        for j in range(3):
            assert float((_lap_component(F[j]) - lapF[j]).norm()) < TOL
