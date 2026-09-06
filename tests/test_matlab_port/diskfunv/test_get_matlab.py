"""Port of MATLAB Chebfun tests/diskfunv/test_get.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_get.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebpref import ChebfunPref
from chebfunjax.diskfun.diskfunv import Diskfunv

from ..diskfun._cart import disk_xy

jax.config.update("jax_enable_x64", True)


class TestDiskfunvGet:
    def test_all_matlab_assertions(self):
        tol = 1e2 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = disk_xy(lambda x, y: 1 + jnp.sin(np.pi * x * y) + jnp.sin(np.pi * x))
        g = disk_xy(lambda x, y: jnp.cos(1 - x * y))
        F = Diskfunv(f, g)
        Fc = F.components
        G = F.T
        assert float((Fc[0] - f).norm()) < tol                               # pass(1)
        assert float((Fc[1] - g).norm()) < tol                               # pass(2)
        assert abs(2 - F.n_components) < tol                                 # pass(3)
        assert F.is_transposed is False                                      # pass(4)
        assert G.is_transposed is True                                       # pass(5)
