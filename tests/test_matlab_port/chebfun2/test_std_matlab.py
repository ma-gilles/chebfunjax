"""Port of MATLAB Chebfun tests/chebfun2/test_std.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_std.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

TOL = 1e3 * float(np.finfo(np.float64).eps)
_T = jnp.asarray(np.linspace(-1.0, 1.0, 101))
_EXACT = float(np.sqrt(4.0 / 45.0))


class TestChebfun2Std:
    def test_default_dim(self):
        # f1 = y^2 + x has y-std sqrt(4/45), a function of x.
        f1 = Chebfun2.from_function(lambda x, y: y**2 + x)
        s1 = f1.std()
        assert float(np.max(np.abs(np.asarray(s1(_T)) - _EXACT))) < TOL

    def test_dim1(self):
        f1 = Chebfun2.from_function(lambda x, y: y**2 + x)
        ss1 = f1.std(None, 1)
        assert float(np.max(np.abs(np.asarray(ss1(_T)) - _EXACT))) < TOL

    def test_dim2(self):
        # f2 = x^2 + y has x-std sqrt(4/45), a function of y.
        f2 = Chebfun2.from_function(lambda x, y: x**2 + y)
        s2 = f2.std(None, 2)
        assert float(np.max(np.abs(np.asarray(s2(_T)) - _EXACT))) < TOL
