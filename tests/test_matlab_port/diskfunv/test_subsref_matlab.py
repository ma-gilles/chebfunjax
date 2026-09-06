"""Port of MATLAB Chebfun tests/diskfunv/test_subsref.m (Fable 5).

``F(1)`` (the first component) is ``F.components[0]``.

Provenance
----------
MATLAB source : tests/diskfunv/test_subsref.m
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


class TestDiskfunvSubsref:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = disk_xy(lambda x, y: jnp.sin(10 * x * y))
        F = Diskfunv(f, f)
        G = F.components[0]
        exact = np.asarray(G.pivot_values)
        assert np.linalg.norm(exact - np.asarray(F.components[0].pivot_values)) < tol  # pass(1)
