"""Retained supplemental analytic mean checks; not original source predicates."""

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
