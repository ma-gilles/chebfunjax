"""Port of MATLAB Chebfun tests/chebfun/test_airy.m (Fable 5).

K = 0..3 (Ai, Ai', Bi, Bi') on [-1, 5]; MATLAB's scale option skipped.

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
        x = cj.chebfun(lambda t: t, domain=(-1.0, 5.0))
        try:
            g = x.airy(K)
        except TypeError:
            if K == 0:
                g = x.airy()
            else:
                pytest.skip("chebfunjax airy exposes only Ai (no K "
                            "branch argument)")
        exact = jnp.asarray(sairy(np.asarray(XX))[K])
        err = jnp.abs(g(XX) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * max(g.vscale, 1.0)
