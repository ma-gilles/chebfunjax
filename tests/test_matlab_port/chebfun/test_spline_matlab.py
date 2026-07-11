"""Port of MATLAB Chebfun tests/chebfun/test_spline.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_spline.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunSpline:
    def test_interpolates_data(self):
        x = jnp.arange(11.0)
        y = jnp.sin(x)
        f = cj.chebfun(lambda t: t, domain=(0.0, 10.0)).spline(x, y)
        err = jnp.abs(f(x) - y)
        assert float(jnp.max(err)) < 100 * EPS
        assert len(f.funs) == 10

    def test_c2_continuity(self):
        # spline's defining property: continuous 2nd derivative at knots
        x = jnp.arange(6.0)
        y = jnp.asarray([0.0, 1.0, -0.5, 2.0, 0.3, 1.0])
        f = cj.chebfun(lambda t: t, domain=(0.0, 5.0)).spline(x, y)
        d2 = f.diff(2)
        for k in [1.0, 2.0, 3.0, 4.0]:
            left = float(d2(jnp.asarray(k - 1e-9)))
            right = float(d2(jnp.asarray(k + 1e-9)))
            assert abs(left - right) < 1e-5

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued chebfun")
