"""Port of MATLAB Chebfun tests/trigtech/test_conj.m (Opus 4.8[1m]).

conj(f) conjugates the Fourier coefficients.

Provenance
----------
MATLAB source : tests/trigtech/test_conj.m
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


def _coeffs_close(h, g, tol):
    n = max(h.n, g.n)
    return _ninf(h.prolong(n).coeffs - g.prolong(n).coeffs) < tol


class TestTrigtechConj:
    def test_scalar(self):
        # f = cos + i sin = e^{i pi x}; conj(f) = e^{-i pi x} = g.
        f = _tt(lambda x: jnp.cos(jnp.pi * x) + 1j * jnp.sin(jnp.pi * x))
        g = _tt(lambda x: jnp.cos(jnp.pi * x) - 1j * jnp.sin(jnp.pi * x))
        h = f.conj()
        assert _coeffs_close(h, g, 10 * h.vscale * EPS)

    def test_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x) + 1j * jnp.sin(jnp.pi * x),
             -jnp.exp(1j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.cos(jnp.pi * x) - 1j * jnp.sin(jnp.pi * x))
        h = f.conj()
        n = max(h.n, g.n)
        gcol = g.prolong(n).coeffs.reshape(-1, 1)
        expected = jnp.concatenate([gcol, -gcol], axis=1)
        assert _ninf(h.prolong(n).coeffs - expected) < 10 * h.vscale * EPS

    def test_mixed_array(self):
        # A real column and a complex column.
        f = _tt(lambda x: jnp.stack(
            [jnp.exp(jnp.cos(jnp.pi * x)), jnp.exp(1j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.stack(
            [jnp.exp(jnp.cos(jnp.pi * x)),
             jnp.cos(jnp.pi * x) - 1j * jnp.sin(jnp.pi * x)], axis=-1))
        h = f.conj()
        assert _coeffs_close(h, g, 10 * h.vscale * EPS)
