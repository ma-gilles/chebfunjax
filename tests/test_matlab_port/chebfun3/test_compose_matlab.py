"""Port of MATLAB Chebfun tests/chebfun3/test_compose.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): the Chebfun3 composition
operators exist.

MATLAB repeats each composition three more times with a 'fiberDim'
constructor flag (pass 3-5, 7-9, ...); that flag is not exposed by the
chebfunjax constructor, so only the flag-free assertion of each group is
ported.

Provenance
----------
MATLAB source : tests/chebfun3/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


def _base():
    return chebfun3(
        lambda x, y, z: jnp.cos(x * y * z) + jnp.sin(x * y * z) + y - 0.1)


class TestChebfun3Compose:
    def test_multiplication(self):
        # pass(1): f .* sin((x-.1)(y+.4)(z+.8)).
        f = _base()
        g = chebfun3(
            lambda x, y, z: (jnp.cos(x * y * z) + jnp.sin(x * y * z)
                             + y - 0.1)
            * jnp.sin((x - 0.1) * (y + 0.4) * (z + 0.8)))
        s = chebfun3(
            lambda x, y, z: jnp.sin((x - 0.1) * (y + 0.4) * (z + 0.8)))
        assert float((g - f * s).norm()) < TOL

    def test_sine(self):
        # pass(2)
        g = chebfun3(
            lambda x, y, z: jnp.sin(jnp.cos(x * y * z)
                                    + jnp.sin(x * y * z) + y - 0.1))
        assert float((g - _base().sin()).norm()) < TOL

    def test_cosine(self):
        # pass(6)
        g = chebfun3(
            lambda x, y, z: jnp.cos(jnp.cos(x * y * z)
                                    + jnp.sin(x * y * z) + y - 0.1))
        assert float((g - _base().cos()).norm()) < TOL

    def test_compose_with_callable(self):
        # compose(f, op) is the general form the operators route through.
        f = _base()
        assert float((f.compose(jnp.sin) - f.sin()).norm()) < TOL
