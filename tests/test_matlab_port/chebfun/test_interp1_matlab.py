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

    # FIXED (Fable 5, Big-Three array-valued epic): interp1 now takes
    # (n, m) y-data column-wise, so the polynomial-mode array case
    # ports (data-hit check at the same 1e4*eps scale as the scalar
    # polynomial test).
    def test_array_valued(self):
        x = jnp.arange(11.0)
        y = jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).interp1(x, y)
        assert f.n_columns == 2
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 1e4 * EPS
