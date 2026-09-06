"""Port of MATLAB Chebfun tests/cheb/test_normal2.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_normal2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import pytest

from chebfunjax import cheb
from chebfunjax.chebpref import ChebfunPref

jax.config.update("jax_enable_x64", True)


class TestChebNormal2:
    def test_all_matlab_assertions(self):
        tol = ChebfunPref().cheb2Prefs.chebfun2eps

        # pass(1): indefinite covariance rejected.
        with pytest.raises(ValueError, match="nonSymPosDef"):
            cheb.normal2([0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])

        # pass(2): non-symmetric covariance rejected.
        with pytest.raises(ValueError, match="nonSymPosDef"):
            cheb.normal2([0.0, 0.0], [[3.0, 1.0], [2.0, 3.0]])

        # pass(3): the density integrates to one.
        p = cheb.normal2([2.0, 6.0], [[4.0, -1.2], [-1.2, 4.0]])
        assert abs(float(p.sum2()) - 1.0) < 1000 * tol
