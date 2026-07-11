"""Port of MATLAB Chebfun tests/fun/test_detectEdge.m (Fable 5).

MATLAB's fun.detectEdge locates a sign-jump inside [0,1]; chebfunjax's
splitting edge-locator (_split_find_edge) is the counterpart.

Provenance
----------
MATLAB source : tests/fun/test_detectEdge.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import _split_find_edge

RNG = np.random.default_rng(13)


class TestFunDetectEdge:
    @pytest.mark.parametrize("trial", range(5))
    def test_locates_sign_jump(self, trial):
        x0 = float(RNG.uniform(0.1, 0.9))

        def f(x):
            return (jnp.exp(x) + jnp.cos(7 * x)
                    + 0.1 * jnp.sign(x - x0))
        edge = float(_split_find_edge(f, 0.0, 1.0, 12))
        assert abs(edge - x0) < 1e-7, f"x0={x0}, edge={edge}"
