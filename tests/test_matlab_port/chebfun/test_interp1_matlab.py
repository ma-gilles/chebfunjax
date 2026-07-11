"""Port of MATLAB Chebfun tests/chebfun/test_interp1.m (Fable 5).

chebfunjax interp1 produces the global polynomial interpolant;
MATLAB's 'linear'/'pchip'/'spline' modes are separate methods (pchip/
spline exist on Chebfun; 'linear' has no counterpart).

Provenance
----------
MATLAB source : tests/chebfun/test_interp1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunInterp1:
    def test_polynomial_interpolant_hits_data(self):
        x = jnp.arange(11.0)
        y = jnp.sin(x)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).interp1(x, y)
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 1e4 * EPS

    def test_linear_mode(self):
        pytest.skip("chebfunjax interp1 has no 'linear' mode")

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued chebfun")
