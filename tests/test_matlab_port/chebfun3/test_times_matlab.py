"""Port of MATLAB Chebfun tests/chebfun3/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e4 * EPS


@pytest.fixture(scope="module")
def f():
    return Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))


class TestChebfun3Times:
    def test_scalar_times(self, f):
        assert maxdiff(f * 2, lambda x, y, z: 2 * jnp.cos(x * y * z)) < TOL
        assert maxdiff(2 * f, lambda x, y, z: 2 * jnp.cos(x * y * z)) < TOL

    def test_square(self, f):
        assert maxdiff(f ** 2,
                       lambda x, y, z: jnp.cos(x * y * z) ** 2) < TOL
        assert maxdiff(f * f,
                       lambda x, y, z: jnp.cos(x * y * z) ** 2) < TOL

    def test_times_general(self, f):
        g = Chebfun3.from_function(lambda x, y, z: x + y + z + x * y * z)
        assert maxdiff(
            f * g,
            lambda x, y, z: jnp.cos(x * y * z) * (x + y + z + x * y * z),
        ) < 10 * TOL
