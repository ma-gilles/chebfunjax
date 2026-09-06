"""Port of MATLAB Chebfun tests/chebfun/test_points.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_points.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.utils.quadrature import chebpts

jax.config.update("jax_enable_x64", True)


class TestChebfunPoints:
    def test_all_matlab_assertions(self):
        n = 10
        tol = 10 * np.finfo(float).eps
        r = jnp.asarray(np.random.RandomState(0).rand(n))
        for k in (1, 2):
            f = chebfun(r, domain=(0.0, 1.0), chebkind=k)
            expected = (np.asarray(chebpts(n, kind=k)) + 1.0) / 2.0
            assert np.linalg.norm(np.asarray(f.points()) - expected) < tol
