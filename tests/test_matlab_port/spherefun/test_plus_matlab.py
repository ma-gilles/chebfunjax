"""Port of MATLAB Chebfun tests/spherefun/test_plus.m (Fable 5).

FIXED: Spherefun arithmetic added in the Fable 5 audit (constructor
re-approximation, MATLAB semantics).

Provenance
----------
MATLAB source : tests/spherefun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.spherefun.spherefun import Spherefun


class TestSpherefunPlus:
    def test_plus_functions_and_scalar(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = Spherefun.from_function(lambda l, t: jnp.cos(t))
            h = f + f
            hs = f + 2.0
        pt = (jnp.asarray(0.7), jnp.asarray(1.1))
        v = np.cos(1.1)
        assert abs(float(h(*pt)) - (v + v)) < 1e-12
        assert abs(float(hs(*pt)) - (v + 2.0)) < 1e-12
