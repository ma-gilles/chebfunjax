"""Port of MATLAB Chebfun tests/chebfun2/test_plus.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e5 * EPS
_NATIVE = json.loads((Path(__file__).parents[2] / "references" /
                      "chebfun2_plus_native.json").read_text())
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
        # MATLAB pass(4): length(f+f) == length(f) for f=x.
        # Python rank is the single-output MATLAB length(f).
        f = Chebfun2.from_function(lambda x, y: x)
        g = f + f
        x = jnp.asarray(0.3)
        y = jnp.asarray(-0.4)
        assert abs(float(g(x, y)) - 2 * float(f(x, y))) < TOL
        assert g.rank == f.rank == 1


    def test_overwritten_cosine_rank_predicate_matches_native_evaluation(self):
        # Native's first pass(5) evaluates false (ranks7/6), then is overwritten.
        # Preserve that evaluation; the returned test array has all8slots true.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = f + f
        assert (f.rank, g.rank) == (_NATIVE["cosine_f_rank"], _NATIVE["cosine_g_rank"])
        assert bool(g.rank == f.rank) == _NATIVE["cosine_evaluated_predicate"]

    def test_small_vertical_scale_addition(self):
        # Native pass(5), overwritten assignment: f=1e-100*x, rank is retained.
        # Native pass(6): abs(vscale(g)-2e-100) < 1e5*cheb2eps.
        f = Chebfun2.from_function(lambda x, y: 1e-100 * x)
        g = f + f
        assert g.rank == f.rank
        assert abs(g.vscale() - 2e-100) < TOL

    def test_large_vertical_scale_addition(self):
        # Native pass(7): f=1e100*x, rank is retained.
        # Native pass(8): relative vscale error is less than 2e-2.
        f = Chebfun2.from_function(lambda x, y: 1e100 * x)
        g = f + f
        assert g.rank == f.rank
        assert abs(g.vscale() - 2e100) / 2e100 < 2e-2
