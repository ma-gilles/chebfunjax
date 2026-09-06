"""Port of MATLAB Chebfun tests/chebfun3t/test_ndf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3t/test_ndf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3t import chebfun3t

jax.config.update("jax_enable_x64", True)


class TestChebfun3tNdf:
    def test_all_matlab_assertions(self):
        assert chebfun3t().ndf() == 0                                        # pass(1)
        assert chebfun3t(lambda x, y, z: 10 + 0 * x).ndf() == 1              # pass(2)
        dom = (-1, 2, -np.pi / 2, np.pi, -3, 1)
        assert chebfun3t(lambda x, y, z: x + 0 * y, dom).ndf() == 2          # pass(3)
        assert chebfun3t(lambda x, y, z: y + 0 * x, dom).ndf() == 2          # pass(4)
        assert chebfun3t(lambda x, y, z: z + 0 * x, dom).ndf() == 2          # pass(5)
        f = chebfun3t(lambda x, y, z: jnp.sin(np.pi * (x + y + z)))
        assert f.ndf() == int(np.prod(f.coeffs.shape))                       # pass(6)
