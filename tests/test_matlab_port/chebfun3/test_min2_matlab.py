"""Port of MATLAB Chebfun tests/chebfun3/test_min2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_min2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e12 * float(np.finfo(np.float64).eps)
_TN = np.linspace(-1.0, 1.0, 101)
_T = jnp.asarray(_TN)


class TestChebfun3Min2:
    """min2(x^2+y^2+z^2) over two dims is the remaining variable squared."""

    def setup_method(self):
        self.f = Chebfun3.from_function(
            lambda x, y, z: x**2 + y**2 + z**2)

    def _err(self, h):
        return float(np.max(np.abs(np.asarray(h(_T)) - _TN**2)))

    def test_default(self):
        assert self._err(self.f.min2()) < TOL

    def test_g_none(self):
        assert self._err(self.f.min2(None)) < TOL

    @pytest.mark.parametrize("dims", [(1, 2), (2, 1), (1, 3), (2, 3)])
    def test_dims(self, dims):
        assert self._err(self.f.min2(None, dims)) < TOL
