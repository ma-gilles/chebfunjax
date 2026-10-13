"""Literal port of MATLAB tests/chebfun3t/test_sum3.m.

Provenance
----------
MATLAB source : tests/chebfun3t/test_sum3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d import chebfun3t

EPS = float(np.finfo(np.float64).eps)
TOL = 1e1 * EPS


@pytest.mark.slow
class TestChebfun3tSum3:
    def test_all_matlab_assertions(self):
        assert np.asarray(chebfun3t().sum3()).size == 0

        f = chebfun3t(lambda x, y, z: jnp.ones_like(x))
        assert abs(float(f.sum3()) - 8.0) < TOL

        f = chebfun3t(lambda x, y, z: 1.0 / (1 + x**2 + y**2 + z**2))
        assert abs(float(f.sum3()) - 4.28685406230184188268) < TOL

        dom = (0.0, 1.0, 0.0, 2.0, 0.0, 3.0)
        fx = chebfun3t(lambda x, y, z: x, dom)
        assert abs(float(fx.sum3()) - 3.0) < TOL
        fy = chebfun3t(lambda x, y, z: y, dom)
        assert abs(float(fy.sum3()) - 6.0) < TOL
        fz = chebfun3t(lambda x, y, z: z, dom)
        assert abs(float(fz.sum3()) - 9.0) < TOL
