"""Port of MATLAB Chebfun tests/chebfun/test_circconv.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_circconv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunCircconv:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    @pytest.mark.xfail(
        reason="chebfunjax circconv: NaN on non-unit domains ([-pi,pi]) "
        "and on [-1,1] the result is circularly shifted by half the "
        "period (sign flip for odd harmonics) with ~0.6% amplitude "
        "error. Two real defects flagged in the Fable 5 audit.")
    def test_circconv_of_cos_eigenfunction(self):
        # circular convolution of cos(k x) with itself on [-pi, pi]
        # gives pi*cos(k x) (Fourier eigen-property).
        f = cj.chebfun(lambda x: jnp.cos(10 * x),
                       domain=(-float(np.pi), float(np.pi)), trig=True)
        h = f.circconv(f)
        xs = jnp.asarray(np.linspace(-3.0, 3.0, 60))
        exact = np.pi * jnp.cos(10 * xs)
        err = jnp.abs(h(xs) - exact)
        assert float(jnp.max(err)) < 1e4 * EPS
