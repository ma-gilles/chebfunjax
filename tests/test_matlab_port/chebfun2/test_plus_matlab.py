"""Port of MATLAB Chebfun tests/chebfun2/test_plus.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e5 * EPS
D = [(-1.0, 1.0, -1.0, 1.0), (-2.0, 2.0, -2.0, 2.0),
     (-1.0, float(np.pi), 0.0, float(2 * np.pi))]


def _norm(f):
    return float(f.norm())


class TestChebfun2Plus:
    @pytest.mark.parametrize("dom", D)
    def test_plus_matches_direct_construction(self, dom):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        g = Chebfun2.from_function(lambda x, y: x + y + x * y, domain=dom)
        fpg = Chebfun2.from_function(
            lambda x, y: jnp.cos(x * y) + (x + y + x * y), domain=dom)
        tolr = float(np.max(np.abs(dom))) * TOL
        assert _norm(f + g - fpg) < tolr

    def test_rank_not_inflated_rank1(self):
        # MATLAB pass(4)-(6): length(f+f) == length(f) after compression.
        # Chebfun2.plus concatenates the CDR factors then recompresses via
        # _compress(), so rank(f+f) stays rank(f) and the values are exact.
        f = Chebfun2.from_function(lambda x, y: x)
        g = f + f
        x = jnp.asarray(0.3)
        y = jnp.asarray(-0.4)
        assert abs(float(g(x, y)) - 2 * float(f(x, y))) < TOL
        assert g.rank == f.rank == 1
