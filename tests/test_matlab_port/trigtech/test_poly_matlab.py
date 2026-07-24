"""Port of MATLAB Chebfun tests/trigtech/test_poly.m (Opus 4.8[1m]).

poly(f) returns the Fourier coefficients as a polynomial-style row (the
transpose of the stored coefficients).

Provenance
----------
MATLAB source : tests/trigtech/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechPoly:
    def test_zeros(self):
        f = _tt(lambda x: jnp.zeros_like(x))
        p = f.poly()
        assert _ninf(p) <= 10 * max(f.vscale, 1.0) * EPS

    def test_constant(self):
        f = _tt(lambda x: 3 * jnp.ones_like(x))
        p = f.poly()
        assert _ninf(p - 3) < 10 * f.vscale * EPS

    def test_one_plus_cos(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        p = f.poly()
        assert _ninf(p - jnp.array([0.5, 1.0, 0.5])) < 10 * f.vscale * EPS

    def test_complex_exponentials(self):
        f = _tt(lambda x: 1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x))
        p = f.poly()
        assert _ninf(p - jnp.array([0, 1, 1, 0, 1])) < 10 * f.vscale * EPS

    def test_array_valued(self):
        f = _tt(lambda x: jnp.stack([
            3 * jnp.ones_like(x),
            1 + jnp.cos(jnp.pi * x),
            1 + jnp.exp(2j * jnp.pi * x) + jnp.exp(-1j * jnp.pi * x),
        ], axis=-1))
        p = f.poly()
        # Rows of p correspond to columns of f (MATLAB p_exact.').
        p_exact = jnp.array([
            [0, 0, 0],
            [0, 0.5, 1],
            [3, 1, 1],
            [0, 0.5, 0],
            [0, 0, 1],
        ], dtype=jnp.complex128).T
        assert _ninf(p - p_exact) < 10 * f.vscale * EPS
