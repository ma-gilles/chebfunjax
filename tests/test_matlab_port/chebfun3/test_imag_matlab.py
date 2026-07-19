"""Port of MATLAB Chebfun tests/chebfun3/test_imag.m (Fable 5).

FIXED (Fable 5): Chebfun3.imag added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff, ninf

TOL = 100 * EPS


class TestChebfun3Imag:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        g = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(x + y ** 2 + z ** 3))
        h = f + 1j * g

        # pass(1): pointwise consistency on a 3x3x3 ndgrid over [-1,1]^3.
        x = np.linspace(-1, 1, 3)
        xx, yy, zz = np.meshgrid(x, x, x, indexing="ij")
        X, Y, Z = (jnp.asarray(xx.ravel()), jnp.asarray(yy.ravel()),
                   jnp.asarray(zz.ravel()))
        assert ninf(jnp.imag(h(X, Y, Z)) - g(X, Y, Z)) < TOL

        # pass(2): imag(h) == g as functions.
        assert maxdiff(h.imag(),
                       lambda x, y, z: jnp.sin(x + y ** 2 + z ** 3)) < TOL
