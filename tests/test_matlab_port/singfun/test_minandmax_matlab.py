"""Port of MATLAB Chebfun tests/singfun/test_minandmax.m (Opus 4.8).

chebfunjax Singfun implements no ``minandmax`` method, so every assertion is
xfailed (the call raises ``AttributeError``).  The Mathematica-derived exact
extrema from the MATLAB test are preserved for when minandmax lands.

Provenance
----------
MATLAB source : tests/singfun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

A = 0.64
B = -0.64
C = 1.28
D = -1.28

_REASON = "chebfunjax Singfun has no minandmax() method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunMinandmax:
    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_root_left_bounded(self):
        f = _sf(lambda x: (1 + x) ** A * jnp.exp(x), (A, 0.0))
        y, x = f.minandmax()
        y_exact = np.array([0.0, 2 ** A * np.exp(1)])
        assert np.allclose(np.asarray(y), y_exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_pole_left_unbounded(self):
        f = _sf(lambda x: (1 + x) ** D * jnp.sin(50 * np.pi * x), (D + 1, 0.0))
        y, x = f.minandmax()
        assert y[1] == np.inf

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_frac_root_right(self):
        f = _sf(lambda x: (1 - x) ** C * jnp.cos(x), (0.0, C))
        y, x = f.minandmax()
        y_exact = np.array([0.0, 1.511345730595468])
        assert np.allclose(np.asarray(y), y_exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_root_at_left_endpoint(self):
        f = _sf(lambda x: (1 - x) ** B * (jnp.exp(x) - np.exp(1)), (0.0, 1 + B))
        y, x = f.minandmax()
        y_exact = np.array([-1.727141310139675, 0.0])
        assert np.allclose(np.asarray(y), y_exact)

    @pytest.mark.xfail(reason=_REASON, strict=True)
    def test_pole_and_root(self):
        f = _sf(lambda x: (1 + x) ** B * jnp.sin(x) * (1 - x) ** C, (B, C))
        y, x = f.minandmax()
        assert y[0] == -np.inf and np.isclose(float(y[1]), 0.1636938399751735)
