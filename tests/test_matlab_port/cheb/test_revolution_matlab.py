"""Port of MATLAB Chebfun tests/cheb/test_revolution.m (Fable 5).

Provenance
----------
MATLAB source : tests/cheb/test_revolution.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import warnings

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax import cheb

jax.config.update("jax_enable_x64", True)


class TestChebRevolution:
    def test_all_matlab_assertions(self):
        # Cone: f(x) = x on [0, 1].
        f = cj.chebfun(lambda x: x, domain=(0.0, 1.0))
        result = cheb.revolution(f)
        assert abs(result["surfaceArea"] - math.pi * math.sqrt(2)) < 1e-15   # pass(1)
        assert abs(result["volume"] - math.pi / 3) < 1e-15                   # pass(2)
        assert abs(result["centroidZ"] - 3.0 / 4) < 1e-15                    # pass(3)
        assert abs(result["momentOfInertia"] - math.pi / 10) < 1e-15         # pass(4)

        # Gabriel's horn variant: f(x) = exp(-x) on [0, Inf).
        f = cj.chebfun(lambda x: jnp.exp(-x), domain=(0.0, jnp.inf))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cheb.revolution(f)
        assert abs(result["surfaceArea"]
                   - math.pi * (math.sqrt(2) + math.asinh(1))) < 1e-12       # pass(5)
        assert abs(result["volume"] - math.pi / 2) < 1e-12                   # pass(6)
        assert abs(result["centroidZ"] - 0.5) < 1e-6                         # pass(7)
        assert abs(result["momentOfInertia"] - math.pi / 8) < 1e-12          # pass(8)
