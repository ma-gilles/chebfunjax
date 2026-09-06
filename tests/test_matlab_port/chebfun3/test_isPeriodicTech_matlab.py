"""Port of MATLAB Chebfun tests/chebfun3/test_isPeriodicTech.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_isPeriodicTech.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

jax.config.update("jax_enable_x64", True)

pi = np.pi


class TestChebfun3IsPeriodicTech:
    def test_all_matlab_assertions(self):
        f = chebfun3(lambda x, y, z: x)
        assert not f.isPeriodicTech()                                        # pass(1)
        f = chebfun3(lambda x, y, z: jnp.cos(pi * x), trig=True)
        assert f.isPeriodicTech()                                            # pass(2)
        f = chebfun3(lambda x, y, z: jnp.exp(1j * pi * x), trig=True)
        assert f.isPeriodicTech()                                            # pass(3)
        f = chebfun3(lambda x, y, z: jnp.cos(x) * jnp.sin(z),
                     (-pi, pi, -pi, pi, -pi, pi), trig=True)
        assert f.isPeriodicTech()                                            # pass(4)
