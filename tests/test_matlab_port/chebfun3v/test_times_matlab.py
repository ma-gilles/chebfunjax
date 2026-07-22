"""Port of MATLAB Chebfun tests/chebfun3v/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3
from chebfunjax.chebfun3d.chebfun3v import Chebfun3v

EPS = 2.220446049250313e-16
TOL = 1e3 * EPS


class TestChebfun3vTimes:
    def test_scalar_multiply_divide_power(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        F = Chebfun3v([f, f])
        G = Chebfun3v([2 * f, 2 * f])
        H = Chebfun3v([f / 2, f / 2])
        K = Chebfun3v([f ** 2, f ** 2])

        assert float((2 * F - G).norm()) < TOL       # 2*F  (mtimes scalar)
        assert float((2 * F - G).norm()) < TOL       # 2.*F (times scalar)
        assert float((F * 2 - G).norm()) < TOL       # F*2
        assert float((F * 2 - G).norm()) < TOL       # F.*2
        assert float((F / 2 - H).norm()) < TOL       # F/2
        assert float((F / 2 - H).norm()) < TOL       # F./2
        assert float((F / 2 - H).norm()) < TOL       # 2.\F  (== F/2)
        assert float((F / 2 - H).norm()) < TOL       # 2\F   (== F/2)
        assert float((F ** 2 - K).norm()) < TOL      # F.^2
        assert float((F * F - K).norm()) < TOL       # F.*F
