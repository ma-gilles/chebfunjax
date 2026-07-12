"""Port of MATLAB Chebfun tests/diskfun/test_norm.m (Fable 5).

FIXED: Diskfun.norm added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/diskfun/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun


class TestDiskfunNorm:
    def test_norm_of_x(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Diskfun.from_function(lambda t, r: r * jnp.cos(t))
        assert abs(float(f.norm()) - np.sqrt(np.pi / 4)) < 1e-12
