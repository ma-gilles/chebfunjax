"""Port of MATLAB Chebfun tests/chebfun/test_deriv.m (Fable 5).

FIXED: Chebfun.deriv added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun/test_deriv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunDeriv:
    def test_derivative_evaluation(self):
        f = cj.chebfun(jnp.sin)
        assert abs(float(f.deriv(0.3)) - np.cos(0.3)) < 1e-12
        assert abs(float(f.deriv(0.3, 2)) + np.sin(0.3)) < 1e-10
