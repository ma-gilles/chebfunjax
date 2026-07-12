"""Port of MATLAB Chebfun tests/spherefun/test_norm.m (Fable 5).

FIXED: Spherefun.norm added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/spherefun/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunNorm:
    def test_norm_of_cos_theta(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Spherefun.from_function(lambda l, t: jnp.cos(t))
        assert abs(float(f.norm()) - np.sqrt(4 * np.pi / 3)) < 1e-12
