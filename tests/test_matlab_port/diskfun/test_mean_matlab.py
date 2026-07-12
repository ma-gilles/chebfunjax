"""Port of MATLAB Chebfun tests/diskfun/test_mean.m (Fable 5).

FIXED: Diskfun.mean added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfun/test_mean.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun


class TestDiskfunMean:
    def test_mean_of_one(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one = Diskfun.from_function(lambda t, r: jnp.ones_like(r))
        assert abs(float(one.mean()) - 1.0) < 1e-12
