"""Port of MATLAB Chebfun tests/chebfun3/test_squeeze.m (Fable 5).

FIXED: Chebfun3.squeeze added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun3/test_squeeze.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e3 * np.finfo(float).eps


class TestChebfun3Squeeze:
    def test_no_squeeze(self):
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x * y * z))
        assert isinstance(f.squeeze(), Chebfun3)

    def test_3d_to_1d(self):
        xs = jnp.asarray(np.linspace(-1, 1, 15))
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x)).squeeze()
        assert isinstance(f, Chebfun)
        assert float(jnp.max(jnp.abs(f(xs) - jnp.cos(xs)))) < TOL

        g = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(y),
            domain=(-1, 1, -2, 3, -1, 1)).squeeze()
        ys = jnp.asarray(np.linspace(-2, 3, 15))
        assert float(jnp.max(jnp.abs(g(ys) - jnp.cos(ys)))) < TOL

        h = Chebfun3.from_function(
            lambda x, y, z: jnp.sin(z),
            domain=(-1, 1, -2, 3, -np.pi, np.pi)).squeeze()
        zs = jnp.asarray(np.linspace(-np.pi, np.pi, 15))
        assert float(jnp.max(jnp.abs(h(zs) - jnp.sin(zs)))) < TOL

    def test_3d_to_2d(self):
        ss = jnp.asarray(np.linspace(-1, 1, 9))
        aa, bb = jnp.meshgrid(ss, ss, indexing="ij")
        f = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(y * z)).squeeze()
        assert isinstance(f, Chebfun2)
        assert float(jnp.max(jnp.abs(f(aa, bb) - jnp.cos(aa * bb)))) \
            < TOL

        g = Chebfun3.from_function(
            lambda x, y, z: jnp.cos(x * z)).squeeze()
        assert isinstance(g, Chebfun2)
        assert float(jnp.max(jnp.abs(g(aa, bb) - jnp.cos(aa * bb)))) \
            < TOL
