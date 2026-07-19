"""Port of MATLAB Chebfun tests/chebfun3/test_conj.m (Fable 5).

FIXED (Fable 5): Chebfun3.conj added in the audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

from ._helpers import EPS, maxdiff

TOL = 100 * EPS


class TestChebfun3Conj:
    def test_all_matlab_assertions(self):
        # pass(1): conjugate of a real-valued Chebfun3 is itself.
        f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))
        h = f.conj()
        assert maxdiff(h, lambda x, y, z: jnp.cos(x * y * z)) < TOL

        # pass(2): conjugate of 1i*f is -1i*f.
        h = (1j * f).conj()
        assert maxdiff(h,
                       lambda x, y, z: -1j * jnp.cos(x * y * z)) < 100 * TOL

        # pass(3): conjugate of f+1i*g is f-1i*g.
        g = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(x + y ** 2 - z ** 3))
        h = (f + 1j * g).conj()
        assert maxdiff(
            h, lambda x, y, z: jnp.cos(x * y * z)
            - 1j * jnp.sin(x + y ** 2 - z ** 3)) < TOL
