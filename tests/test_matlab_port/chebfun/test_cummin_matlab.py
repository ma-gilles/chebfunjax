"""Port of MATLAB Chebfun tests/chebfun/test_cummin.m (Fable 5).

FIXED: Chebfun.cummin added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_cummin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunCummin:
    def test_running_min_of_cos(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(jnp.cos, domain=(0.0, 4.71))
            cm = f.cummin()
        assert abs(float(cm(jnp.asarray(4.5))) + 1.0) < 1e-6
        assert abs(float(cm(jnp.asarray(1.0))) - np.cos(1.0)) < 1e-6
