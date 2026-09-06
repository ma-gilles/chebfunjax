"""Port of MATLAB Chebfun tests/chebfun3/test_domainChck.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_domainChck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun3d.chebfun3 import chebfun3, domainCheck

jax.config.update("jax_enable_x64", True)


class TestChebfun3DomainChck:
    def test_all_matlab_assertions(self):
        ff = lambda x, y, z: jnp.cos(x + y + z)  # noqa: E731
        gg = lambda x, y, z: jnp.sin(x * y * z)  # noqa: E731
        assert domainCheck(chebfun3(ff), chebfun3(gg))                       # pass(1)
        dom = (-1, 2, -2, 1, -3, 0)
        assert domainCheck(chebfun3(ff, dom), chebfun3(gg, dom))             # pass(2)
