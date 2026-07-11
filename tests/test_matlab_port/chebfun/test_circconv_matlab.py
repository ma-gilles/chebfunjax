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

    def test_circconv_of_cos_eigenfunction(self):
        # FIXED (Fable 5): g is now sampled periodically at m*dx (was
        # offset by 'a' -> half-period shift) and the result is rebuilt
        # as a Fourier series (was Runge-diverging polynomial interp).
        # circular convolution of cos(k x) with itself on [-pi, pi]
        # gives pi*cos(k x) (Fourier eigen-property).
        f = cj.chebfun(lambda x: jnp.cos(10 * x),
                       domain=(-float(np.pi), float(np.pi)), trig=True)
        h = f.circconv(f)
        xs = jnp.asarray(np.linspace(-3.0, 3.0, 60))
        exact = np.pi * jnp.cos(10 * xs)
        err = jnp.abs(h(xs) - exact)
        assert float(jnp.max(err)) < 1e4 * EPS

    def test_asymmetric_domain_vs_quadrature(self):
        from scipy.integrate import quad
        k = cj.chebfun(lambda x: jnp.sin(jnp.pi * x),
                       domain=(0.0, 2.0), trig=True)
        h = k.circconv(k)
        for xv in (0.3, 0.9, 1.6):
            ref = quad(lambda t, xv=xv: np.sin(np.pi * t)
                       * np.sin(np.pi * ((xv - t) % 2.0)),
                       0, 2, limit=200)[0]
            assert abs(float(h(jnp.asarray(xv))) - ref) < 1e-10
