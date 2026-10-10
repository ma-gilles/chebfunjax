"""All five native abs predicates, tests/chebfun3/test_abs.m,7574c77.
Original functions, domains, continuous norms and1000*prefeps preserved.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 1000 * ChebfunPref().cheb3Prefs.chebfun3eps


class TestChebfun3Abs:
    def test_positive_default_domain(self):
        # pass(1,2): abs(f) == f and abs(-f) == f for f = cos(xyz) + 2 > 0.
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z) + 2)
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL

    def test_positive_shifted_domain(self):
        # pass(3,4): same on [-3 4 -1 1 -2 0].
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z) + 2,
                                   domain=(-3, 4, -1, 1, -2, 0))
        assert float((f - f.abs()).norm()) < TOL
        assert float((f - (-f).abs()).norm()) < TOL

    def test_sign_change_errors(self):
        f = chebfun3(lambda x, y, z: .9*x+y**2)
        with pytest.raises(ValueError, match='CHEBFUN:CHEBFUN3:abs:notSmooth'):
            f.abs()
