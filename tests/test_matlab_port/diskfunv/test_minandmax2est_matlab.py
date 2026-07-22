"""Port of MATLAB Chebfun tests/diskfunv/test_minandmax2est.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_minandmax2est.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfunv import Diskfunv

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


def _is_subset(rng, box, tol):
    return all(rng[i] >= box[i] - tol for i in (0, 2)) and all(
        rng[i] <= box[i] + tol for i in (1, 3)
    )


class TestDiskfunvMinandmax2est:
    def test_range_subset(self):
        # pass(1): F = [x, y]; isSubset(minandmax2est(F), [-1,1,-1,1], tol)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            F = Diskfunv.from_functions(
                lambda t, r: r * jnp.cos(t), lambda t, r: r * jnp.sin(t)
            )
        rng = np.asarray(F.minandmax2est())
        assert rng.shape == (4,)
        assert _is_subset(rng, (-1.0, 1.0, -1.0, 1.0), _TOL)
