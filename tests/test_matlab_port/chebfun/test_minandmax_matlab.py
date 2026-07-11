"""Port of MATLAB Chebfun tests/chebfun/test_minandmax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
Y_EXACT = np.array([0.710869767377087, 1.884217141925336])


def _f(x):
    return ((x - 0.2) ** 3 - (x - 0.2) + 1) / jnp.cos(x - 0.2)


class TestChebfunMinandmax:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_smooth_reference_values(self):
        f = cj.chebfun(_f)
        (xmin, fmin), (xmax, fmax) = f.minandmax()
        y = np.array([float(fmin), float(fmax)])
        assert float(np.max(np.abs(y - Y_EXACT))) <= 100 * f.vscale * EPS
        fx = np.asarray(f(jnp.asarray(np.array([float(xmin),
                                                float(xmax)]))))
        assert float(np.max(np.abs(fx - Y_EXACT))) \
            <= 100 * f.vscale * EPS

    def test_piecewise_same_function(self):
        f = cj.chebfun(_f, domain=list(np.linspace(-1, 1, 10)))
        (xmin, fmin), (xmax, fmax) = f.minandmax()
        y = np.array([float(fmin), float(fmax)])
        assert float(np.max(np.abs(y - Y_EXACT))) <= 1e3 * f.vscale * EPS

    def test_complex_piecewise(self):
        pytest.skip("minandmax of a complex chebfun (by magnitude) not "
                    "implemented")
