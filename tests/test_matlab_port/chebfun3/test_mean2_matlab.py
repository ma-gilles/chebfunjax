"""Port of MATLAB Chebfun tests/chebfun3/test_mean2.m (Fable 5).

FIXED: Chebfun3 per-dimension reductions (sum/mean/sum2/mean2)
added in the Fable 5 audit (Gauss quadrature over the reduced
variable, re-approximated in the survivors).

Provenance
----------
MATLAB source : tests/chebfun3/test_mean2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

YS = jnp.asarray(np.linspace(-1, 1, 9))
YY, ZZ = jnp.meshgrid(YS, YS, indexing="ij")
TOL = 1e-12


class TestChebfun3Mean2:
    def test_mean_over_two_dims(self):
        f = Chebfun3.from_function(lambda x, y, z: x ** 2 + y * z)
        m = f.mean2((1, 2))   # -> constant 1/3 in z
        zs = jnp.asarray(np.linspace(-1, 1, 9))
        assert float(jnp.max(jnp.abs(m(zs) - 1 / 3))) < TOL
        m3 = f.mean2((2, 3))  # -> x^2
        assert float(jnp.max(jnp.abs(m3(zs) - zs ** 2))) < TOL
