"""Port of MATLAB Chebfun tests/chebfun3/test_mean3.m (Fable 5).

FIXED: Chebfun3.mean3 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_mean3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebpref import ChebfunPref

TOL = 100 * ChebfunPref().cheb3Prefs.chebfun3eps


class TestChebfun3Mean3:
    def test_native_mean3_x2_y4_exp_z(self):
        # tests/chebfun3/test_mean3.m: pass(1)
        f = Chebfun3.from_function(lambda x, y, z: x**2 * y**4 * jnp.exp(z))
        exact = (jnp.exp(1.0) - jnp.exp(-1.0)) / 30.0
        assert abs(f.mean3() - exact) < TOL

    def test_native_mean3_sine_sum(self):
        # tests/chebfun3/test_mean3.m: pass(2)
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(jnp.pi * x)**2
            + jnp.sin(jnp.pi * (y + z))**2
        )
        assert abs(f.mean3() - 1.0) < TOL

    def test_mean3_of_r2(self):
        f = Chebfun3.from_function(lambda x, y, z: x * x + y * y + z * z)
        assert abs(float(f.mean3()) - 1.0) < 1e-12
