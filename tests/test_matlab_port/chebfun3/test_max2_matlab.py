"""Port of MATLAB Chebfun tests/chebfun3/test_max2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_max2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e7 * float(np.finfo(np.float64).eps)
_T = jnp.asarray(np.linspace(-1.0, 1.0, 101))


class TestChebfun3Max2:
    """max2(cos(xyz)) over any two dims is the constant 1."""

    def setup_method(self):
        self.f = Chebfun3.from_function(lambda x, y, z: jnp.cos(x * y * z))

    def _err_vs_one(self, h):
        return float(np.max(np.abs(np.asarray(h(_T)) - 1.0)))

    def test_default(self):
        assert self._err_vs_one(self.f.max2()) < TOL

    def test_g_none(self):
        assert self._err_vs_one(self.f.max2(None)) < TOL

    @pytest.mark.parametrize(
        "dims", [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)])
    def test_dims(self, dims):
        assert self._err_vs_one(self.f.max2(None, dims)) < TOL
