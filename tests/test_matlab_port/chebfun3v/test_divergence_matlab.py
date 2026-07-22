"""Port of MATLAB Chebfun tests/chebfun3v/test_divergence.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_divergence.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 100 * EPS


class TestChebfun3vDivergence:
    def test_definition(self):
        F = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x),
                                     lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.exp(z))
        f1, f2, f3 = F.components
        divF = f1.diff(1) + f2.diff(2) + f3.diff(3)
        assert float((divF - F.divergence()).norm()) < TOL

    def test_divergence_of_gradient_is_laplacian(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        assert float((f.laplacian()
                      - Chebfun3v.gradient(f).divergence()).norm()) < TOL

    def test_two_components(self):
        F = Chebfun3v.from_functions(
            lambda x, y, z: jnp.cos(x),
            lambda x, y, z: jnp.sin(y) * jnp.exp(z))
        divF = Chebfun3.from_function(
            lambda x, y, z: -jnp.sin(x) + jnp.cos(y) * jnp.exp(z))
        assert float((divF - F.divergence()).norm()) < TOL
