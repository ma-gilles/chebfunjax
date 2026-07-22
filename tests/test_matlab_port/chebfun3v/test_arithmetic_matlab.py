"""Port of MATLAB Chebfun tests/chebfun3v/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e3 * EPS


class TestChebfun3vArithmetic:
    def test_plus_minus_times(self):
        f = Chebfun3v.from_functions(lambda x, y, z: jnp.cos(x * y * z),
                                     lambda x, y, z: jnp.cos(x * y * z))
        g = Chebfun3v.from_functions(lambda x, y, z: jnp.sin(y),
                                     lambda x, y, z: jnp.sin(y))

        plus_exact = Chebfun3v.from_functions(
            lambda x, y, z: jnp.cos(x * y * z) + jnp.sin(y),
            lambda x, y, z: jnp.cos(x * y * z) + jnp.sin(y))
        minus_exact = Chebfun3v.from_functions(
            lambda x, y, z: jnp.cos(x * y * z) - jnp.sin(y),
            lambda x, y, z: jnp.cos(x * y * z) - jnp.sin(y))
        mult_exact = Chebfun3v.from_functions(
            lambda x, y, z: jnp.cos(x * y * z) * jnp.sin(y),
            lambda x, y, z: jnp.cos(x * y * z) * jnp.sin(y))

        assert float((f + g - plus_exact).norm()) < TOL
        assert float((f - g - minus_exact).norm()) < TOL
        assert float((f * g - mult_exact).norm()) < TOL
