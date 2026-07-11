"""Port of MATLAB Chebfun tests/diskfun/test_plus.m (Fable 5).

FIXED: Diskfun arithmetic added in the Fable 5 audit (constructor
re-approximation, MATLAB semantics).

Provenance
----------
MATLAB source : tests/diskfun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun


class TestDiskfunPlus:
    def test_plus_functions_and_scalar(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
            h = f + f
            hs = f + 2.0
        pt = (jnp.asarray(0.6), jnp.asarray(0.7))
        v = 0.7 * np.cos(0.6)
        assert abs(float(h(*pt)) - (v + v)) < 1e-12
        assert abs(float(hs(*pt)) - (v + 2.0)) < 1e-12
