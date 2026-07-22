"""Port of MATLAB Chebfun tests/chebfun3v/test_quiver3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_quiver3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3v import Chebfun3v


class TestChebfun3vQuiver3:
    def test_quiver3_does_not_crash(self):
        # MATLAB test only checks that quiver3 runs without error.  The
        # chebfunjax sampler returns the (X, Y, Z, U, V, W) grids a quiver
        # plot would draw.
        F = Chebfun3v.from_functions(lambda x, y, z: x,
                                     lambda x, y, z: y,
                                     lambda x, y, z: z)
        out = F.quiver3()
        assert len(out) == 6
        for arr in out:
            assert jnp.asarray(arr).shape == (9, 9, 9)
