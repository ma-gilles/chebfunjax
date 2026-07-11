"""Port of MATLAB Chebfun tests/chebfun/test_cummax.m (Fable 5).

FIXED: Chebfun.cummax added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_cummax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunCummax:
    def test_running_max_of_sin(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(jnp.sin, domain=(0.0, 4.71))
            cm = f.cummax()
        assert abs(float(cm(jnp.asarray(3.0))) - 1.0) < 1e-6
        assert abs(float(cm(jnp.asarray(1.0))) - np.sin(1.0)) < 1e-6
