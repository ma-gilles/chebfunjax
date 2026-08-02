"""Port of MATLAB Chebfun tests/chebfun/test_join.m (Fable 5).

FIXED: Chebfun.join added in the Fable 5 audit; 2026-08 rewritten to
MATLAB semantics — non-matching domains are translated to be
contiguous (test_join.m pass(6)), never an error.

Provenance
----------
MATLAB source : tests/chebfun/test_join.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj


class TestChebfunJoin:
    def test_adjacent_join(self):
        # pass(1): scalar chebfuns with interior breakpoints
        f = cj.chebfun(jnp.sin, domain=(-1.0, -0.5, 0.0))
        g = cj.chebfun(jnp.sin, domain=(0.0, 0.5, 1.0))
        h = f.join(g)
        xr = np.linspace(-0.99, 0.99, 201)
        assert np.max(np.abs(np.asarray(h(xr)) - np.sin(xr))) < 1e-13

    def test_non_matching_domains_translated(self):
        # pass(6): the domain of g is translated to begin where f ends
        f = cj.chebfun(jnp.sin, domain=(-1.0, -0.5, 0.0))
        g = cj.chebfun(jnp.cos, domain=(1.0, 1.5, 2.0))
        h = f.join(g)
        bps = [float(v) for v in h.domain.breakpoints]
        assert np.allclose(bps, [-1.0, -0.5, 0.0, 0.5, 1.0])
        xl = np.linspace(-1.0, -0.01, 50)
        xr = np.linspace(0.01, 0.99, 50)
        assert np.max(np.abs(np.asarray(h(xl)) - np.sin(xl))) < 1e-13
        assert np.max(np.abs(np.asarray(h(xr)) - np.cos(xr + 1))) < 1e-13

    def test_multi_arg_square_path(self):
        # ConformalVis unit-square path: values retained per segment
        s = cj.chebfun(lambda x: x)
        u = (-1j + s).join(1 + 1j * s, 1j - s, -1 - 1j * s)
        assert float(u.domain.a) == -1.0 and float(u.domain.b) == 7.0
        for t, want in [(-1.0, -1 - 1j), (1.0, 1 - 1j),
                        (3.0, 1 + 1j), (5.0, -1 + 1j), (7.0, -1 - 1j)]:
            assert abs(complex(np.asarray(u(t))) - want) < 1e-13
