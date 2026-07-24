"""Port of MATLAB Chebfun tests/trigtech/test_flipud.m (Opus 4.8[1m]).

flipud(f) maps f(x) -> f(-x) (reverses the domain).

Provenance
----------
MATLAB source : tests/trigtech/test_flipud.m
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


class TestTrigtechFlipud:
    def test_scalar(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        g = _tt(lambda x: -jnp.sin(jnp.pi * x))
        h = f.flipud()
        assert _ninf(g.coeffs - h.coeffs) < 10 * h.vscale * EPS

    def test_array(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.sin(jnp.sin(jnp.pi * x)), jnp.exp(1j * jnp.pi * x)], axis=-1))
        g = _tt(lambda x: jnp.stack(
            [-jnp.sin(jnp.sin(jnp.pi * x)), jnp.exp(-1j * jnp.pi * x)],
            axis=-1))
        h = f.flipud()
        n = max(len(g), len(h))
        assert _ninf(g.prolong(n).coeffs - h.prolong(n).coeffs) \
            < 100 * h.vscale * EPS

    def test_even_length(self):
        f = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x),
             jnp.exp(8j * jnp.pi * x) - jnp.exp(7j * jnp.pi * x)], axis=-1))
        m = 2 * int(np.ceil(f.size(1) / 2))
        f = f.prolong(m)
        g = _tt(lambda x: jnp.stack(
            [jnp.cos(jnp.pi * x),
             jnp.exp(-8j * jnp.pi * x) - jnp.exp(-7j * jnp.pi * x)], axis=-1))
        g = g.prolong(m)
        h = f.flipud()
        assert _ninf(g.coeffs - h.coeffs) < 100 * h.vscale * EPS
