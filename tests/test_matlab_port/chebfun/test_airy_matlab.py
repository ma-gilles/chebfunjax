"""Port of MATLAB Chebfun tests/chebfun/test_airy.m (Fable 5).

K = 0..3 (Ai, Ai', Bi, Bi') on [-1, 5]; MATLAB's scale option and the
complex-argument (1+1i)x sweep are skipped.

FIXED (Fable 5 audit): ``Chebfun.airy(K)`` exposes all four branches, so the
K branch is exercised directly (the earlier conditional skip is now dead).

Provenance
----------
MATLAB source : tests/chebfun/test_airy.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import airy as sairy

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
XX = jnp.asarray(np.linspace(-1, 5, 100))


class TestChebfunAiry:
    @pytest.mark.parametrize("K", [0, 1, 2, 3])
    def test_airy_branches(self, K):
        # pass(1,k): airy(K, x) for K = 0..3 (Ai, Ai', Bi, Bi').
        # FIXED (Fable 5 audit): all four branches exposed via airy(K).
        x = cj.chebfun(lambda t: t, domain=(-1.0, 5.0))
        g = x.airy(K)
        exact = jnp.asarray(sairy(np.asarray(XX))[K])
        err = jnp.abs(g(XX) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)
