"""Port of MATLAB Chebfun tests/trigtech/test_imag.m (Opus 4.8[1m]).

imag(f) extracts the imaginary part of a trigtech.

Provenance
----------
MATLAB source : tests/trigtech/test_imag.m
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


class TestTrigtechImag:
    def test_scalar(self):
        # imag(cos + i sin) = sin.
        f = _tt(lambda x: jnp.cos(jnp.pi * x) + 1j * jnp.sin(jnp.pi * x))
        g = _tt(lambda x: jnp.sin(jnp.pi * x))
        h = f.imag()
        g = g.prolong(len(h))
        assert _ninf(h.coeffs - g.coeffs) < 10 * h.vscale * EPS

    def test_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.sin(jnp.pi * x)) + 1j * jnp.sin(jnp.cos(jnp.pi * x)),
             -jnp.exp(1j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.cos(jnp.pi * x)),
             -jnp.imag(jnp.exp(1j * jnp.pi * x))], axis=-1))
        h = f.imag()
        n = max(len(g), len(h))
        assert _ninf(h.prolong(n).coeffs - g.prolong(n).coeffs) \
            < 100 * h.vscale * EPS

    def test_real_function(self):
        # imag of a real function is a zero (single-coefficient) tech.
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        g = f.imag()
        assert g.coeffs.size == 1

    def test_real_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x), jnp.sin(jnp.pi * x)], axis=-1))
        g = f.imag()
        assert g.coeffs.shape == (1, 2) and _ninf(g.coeffs) == 0
