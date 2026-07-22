"""Port of MATLAB Chebfun tests/chebfun3/test_gradient.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_gradient.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e4 * EPS


class TestChebfun3Gradient:
    def test_linear(self):
        f = Chebfun3.from_function(lambda x, y, z: x)
        exact = Chebfun3v.from_functions(lambda x, y, z: 1 + 0 * x,
                                         lambda x, y, z: 0 * x,
                                         lambda x, y, z: 0 * x)
        assert float((exact - Chebfun3v.gradient(f)).norm()) < TOL

    def test_separable_trig(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z))
        exact = Chebfun3v.from_functions(
            lambda x, y, z: -jnp.sin(x) * jnp.exp(y) * jnp.sin(z),
            lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.sin(z),
            lambda x, y, z: jnp.cos(x) * jnp.exp(y) * jnp.cos(z))
        assert float((exact - Chebfun3v.gradient(f)).norm()) < TOL

    def test_coupled(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        exact = Chebfun3v.from_functions(
            lambda x, y, z: -y * z * jnp.sin(x * y * z),
            lambda x, y, z: -x * z * jnp.sin(x * y * z),
            lambda x, y, z: -x * y * jnp.sin(x * y * z))
        assert float((exact - Chebfun3v.gradient(f)).norm()) < TOL

    def test_polynomial(self):
        f = Chebfun3.from_function(
            lambda x, y, z: x ** 2 + x * y ** 2 * z ** 3)
        exact = Chebfun3v.from_functions(
            lambda x, y, z: 2 * x + y ** 2 * z ** 3,
            lambda x, y, z: 2 * x * y * z ** 3,
            lambda x, y, z: 3 * x * y ** 2 * z ** 2)
        assert float((exact - Chebfun3v.gradient(f)).norm()) < TOL
