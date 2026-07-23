"""Port of MATLAB Chebfun tests/chebfun3/test_min.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

TOL = 1e11 * float(np.finfo(np.float64).eps)
_T = np.linspace(-1.0, 1.0, 41)
_A, _B = np.meshgrid(_T, _T)
_JA, _JB = jnp.asarray(_A), jnp.asarray(_B)
_EXACT = _A**2 + _B**2  # min over one variable of x^2+y^2+z^2


class TestChebfun3Min:
    def setup_method(self):
        self.f = Chebfun3.from_function(
            lambda x, y, z: x**2 + y**2 + z**2)

    def _err(self, h):
        return float(np.max(np.abs(np.asarray(h(_JA, _JB)) - _EXACT)))

    def test_default(self):
        assert self._err(self.f.min()) < TOL

    def test_g_none(self):
        assert self._err(self.f.min(None)) < TOL

    def test_dim2(self):
        assert self._err(self.f.min(None, 2)) < TOL

    def test_dim3(self):
        assert self._err(self.f.min(None, 3)) < TOL

    def test_dim4_returns_f(self):
        assert self.f.min(None, 4) is self.f
