"""Port of MATLAB Chebfun tests/chebfun3/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 1e4 * EPS


class TestChebfun3Mtimes:
    def test_scalar_mtimes(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        assert maxdiff(3 * f,
                       lambda x, y, z: 3 * jnp.cos(x * y * z)) < TOL
        assert maxdiff(f * 3,
                       lambda x, y, z: 3 * jnp.cos(x * y * z)) < TOL

    def test_chebfun3_times_chebfun3v(self):
        pytest.skip("chebfun3 * chebfun3v mtimes needs chebfun3v "
                    "arithmetic (absent)")
