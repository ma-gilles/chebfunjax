"""Port of MATLAB Chebfun tests/chebfun3/test_minandmax3est.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_minandmax3est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebfun3Minandmax3est:
    def test_all_matlab_assertions(self):
        tol = 1000 * ChebfunPref().cheb3Prefs.chebfun3eps
        f = chebfun3(lambda x, y, z: x, (-2, 4, -1, 1, -1, 1))
        mM = np.asarray(f.minandmax3est())
        assert mM.size == 2                                                  # pass(1)
        assert np.linalg.norm(mM - np.array([-2.0, 4.0])) < tol              # pass(2)
