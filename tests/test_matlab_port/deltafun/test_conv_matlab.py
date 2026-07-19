"""Port of MATLAB Chebfun tests/deltafun/test_conv.m (Opus 4.8).

Exercises :meth:`chebfunjax.fun.deltafun.Deltafun.conv` (convolution of
Deltafun objects) against the MATLAB reference assertions.

Provenance
----------
MATLAB source : tests/deltafun/test_conv.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

# pref.deltaPrefs.deltaTol
TOL = 1e-9


def _max_abs_on_grid(fun) -> float:
    """max |fun(x)| over a fine grid of the fun's own domain."""
    a, b = float(fun.domain.a), float(fun.domain.b)
    xs = jnp.asarray(np.linspace(a, b, 200), dtype=jnp.float64)
    return float(jnp.max(jnp.abs(fun(xs))))


class TestDeltafunConv:
    def _build(self):
        dom = Domain((-1.0, 1.0))
        d = Deltafun.empty()
        f = Bndfun.from_function(lambda x: x, dom)
        d1 = Deltafun.from_fun(f)
        d2 = Deltafun(
            Bndfun.from_function(lambda x: jnp.zeros_like(x), dom),
            delta_locs=[0.0],
            delta_mags=[[1.0]],
        )
        return d, f, d1, d2

    def test_conv_empty(self):
        # pass(1): isempty(conv(d,d)) && isempty(conv(d,d1)) && isempty(conv(d1,d))
        d, f, d1, d2 = self._build()
        assert len(d.conv(d)) == 0
        assert len(d.conv(d1)) == 0
        assert len(d1.conv(d)) == 0

    def test_conv_delta_with_smooth(self):
        # pass(2): conv(d1, d2) recovers f = @(x) x within deltaTol
        d, f, d1, d2 = self._build()
        g = d1.conv(d2)
        g0 = g[0]
        # g0 - f should be the zero function on [-1, 1].
        diff = g0 - f
        assert _max_abs_on_grid(diff) < TOL

    def test_conv_delta_with_delta_prime(self):
        # pass(3): conv(d1, diff(d2)) recovers 1 within deltaTol
        d, f, d1, d2 = self._build()
        g = d1.conv(d2.diff(1))
        g0 = g[0]
        # g0 - 1 should be the zero function on [-1, 1].
        diff = g0 - 1.0
        assert _max_abs_on_grid(diff) < TOL
