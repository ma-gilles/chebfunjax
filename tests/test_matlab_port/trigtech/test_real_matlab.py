"""Port of MATLAB Chebfun tests/trigtech/test_real.m (Opus 4.8[1m]).

real(f) extracts the real part of a trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_real.m
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


class TestTrigtechReal:
    def test_scalar(self):
        # real(exp(20i pi x) + 1i sin(100 pi x)) = cos(20 pi x).
        f = _tt(lambda x: jnp.exp(20j * jnp.pi * x)
                + 1j * jnp.sin(100 * jnp.pi * x))
        g = _tt(lambda x: jnp.cos(20 * jnp.pi * x))
        h = f.real()
        g = g.prolong(len(h))
        assert _ninf(h.coeffs - g.coeffs) < 10 * h.vscale * EPS

    def test_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.exp(20j * jnp.pi * x) + 1j * jnp.sin(100 * jnp.pi * x),
             -jnp.exp(10j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.stack(
            [jnp.cos(20 * jnp.pi * x),
             -jnp.real(jnp.exp(10j * jnp.pi * x))], axis=-1))
        h = f.real()
        n = max(len(g), len(h))
        assert _ninf(h.prolong(n).coeffs - g.prolong(n).coeffs) \
            < 10 * h.vscale * EPS

    def test_imaginary_function(self):
        # A purely imaginary function has zero real part.
        f = 1j * _tt(lambda x: jnp.cos(30 * jnp.pi * x))
        g = f.real()
        assert g.coeffs.size == 1 and _ninf(g.coeffs) == 0

    def test_imaginary_array(self):
        f = 1j * _tt(lambda x: jnp.stack(
            [jnp.cos(99 * jnp.pi * x), jnp.sin(99 * jnp.pi * x),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1))
        g = f.real()
        assert g.coeffs.shape == (1, 3) and _ninf(g.coeffs) == 0
