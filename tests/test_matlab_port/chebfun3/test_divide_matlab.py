"""Port of MATLAB Chebfun tests/chebfun3/test_divide.m (Fable 5).

Scalar and root-free function division are ported; divisions creating
singularities are skipped (no blowup support on Chebfun3).

Provenance
----------
MATLAB source : tests/chebfun3/test_divide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e4 * EPS


class TestChebfun3Divide:
    def test_divide_by_scalar(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        assert maxdiff(f / 2,
                       lambda x, y, z: jnp.cos(x * y * z) / 2) < TOL

    def test_divide_by_rootfree_function(self):
        f = Chebfun3.from_function(lambda x, y, z: x * y * z)
        g = Chebfun3.from_function(lambda x, y, z: 2 + jnp.cos(x * y * z))
        assert maxdiff(
            f / g,
            lambda x, y, z: x * y * z / (2 + jnp.cos(x * y * z)),
        ) < 100 * TOL

    def test_divide_creating_singularity(self):
        pytest.skip("division by a chebfun3 with roots requires blowup "
                    "support (absent)")
