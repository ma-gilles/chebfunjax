"""Port of MATLAB Chebfun tests/chebfun3/test_mean.m (Fable 5).

FIXED: Chebfun3 per-dimension reductions (sum/mean/sum2/mean2)
added in the Fable 5 audit (Gauss quadrature over the reduced
variable, re-approximated in the survivors).

Provenance
----------
MATLAB source : tests/chebfun3/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

YS = jnp.asarray(np.linspace(-1, 1, 9))
YY, ZZ = jnp.meshgrid(YS, YS, indexing="ij")
TOL = 1e-12


class TestChebfun3Mean:
    def test_mean_over_each_dim(self):
        f = Chebfun3.from_function(lambda x, y, z: x ** 2 + y * z)
        m2 = f.mean(2)   # mean over y -> x^2
        assert float(jnp.max(jnp.abs(m2(YY, ZZ) - YY ** 2))) < TOL
        m1 = f.mean(1)   # mean over x -> 1/3 + y z
        assert float(jnp.max(jnp.abs(
            m1(YY, ZZ) - (1 / 3 + YY * ZZ)))) < TOL
