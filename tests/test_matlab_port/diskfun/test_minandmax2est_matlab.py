"""Port of MATLAB Chebfun tests/diskfun/test_minandmax2est.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


def _is_subset(mM, box, tol):
    # [mM[0], mM[1]] subset of [box[0], box[1]] within tol.
    return (mM[0] >= box[0] - tol) and (mM[1] <= box[1] + tol)


class TestDiskfunMinandmax2est:
    def test_x_range(self):
        # pass(1): f = x; mM = minandmax2est(f); isSubset(mM, [-1, 1], tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
        mM = np.asarray(f.minandmax2est())
        assert mM.shape == (2,)
        assert _is_subset(mM, (-1.0, 1.0), _TOL)
        # A meaningful estimate: near +/-1 for f = x on the disk.
        assert mM[0] < -0.9 and mM[1] > 0.9
