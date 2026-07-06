"""Port of MATLAB Chebfun tests/deltafun/test_diff.m (Opus 4.8).

Distributional differentiation: ``diff(d, k)`` differentiates the funPart k
times and shifts each delta down k derivative orders (prepends k zero rows to
the magnitude matrix).  MATLAB's ``isequal`` check is reproduced by direct
field comparison, since chebfunjax Deltafun has no ``isequal`` method.

Provenance
----------
MATLAB source : tests/deltafun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

A, B = -4.0, 4.0
DAB = Domain((A, B))
X = jnp.asarray(np.linspace(A, B, 60))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestDeltafunDiff:
    @pytest.mark.skip(
        reason="chebfunjax has no empty Deltafun, so diff(deltafun()) and its "
        "isempty result have no analog"
    )
    def test_diff_empty(self):
        # pass(1): isempty(diff(d)) && isempty(diff(d,4)) for d = deltafun()
        pass

    def test_diff_k0_is_identity(self):
        # pass(2): isequal(d, diff(d,0))
        f = Bndfun.from_function(jnp.sin, DAB)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        d0 = d.diff(0)
        assert _ninf(d0.delta_locs - d.delta_locs) == 0.0
        assert d0.delta_mags.shape == d.delta_mags.shape
        assert _ninf(d0.delta_mags - d.delta_mags) == 0.0
        assert _ninf(d0(X) - d(X)) == 0.0

    def test_diff4_funpart(self):
        # pass(3): iszero(diff(f,4) - dp4.funPart)
        f = Bndfun.from_function(jnp.exp, DAB)
        mag = np.random.rand(4, 4)
        mag[3, 3] = 1.0
        loc = np.sort(np.random.rand(4))
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        dp4 = d.diff(4)
        assert _ninf(f.diff(4)(X) - dp4.funPart(X)) == 0.0

    def test_diff4_locations_unchanged(self):
        # pass(4): norm(dp4.deltaLoc - loc, inf) == 0
        f = Bndfun.from_function(jnp.exp, DAB)
        mag = np.random.rand(4, 4)
        mag[3, 3] = 1.0
        loc = np.sort(np.random.rand(4))
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        dp4 = d.diff(4)
        assert _ninf(dp4.delta_locs - loc) == 0.0

    def test_diff4_magnitude_shift(self):
        # pass(5): [zeros(4,4); mag] - dp4.deltaMag == 0
        f = Bndfun.from_function(jnp.exp, DAB)
        mag = np.random.rand(4, 4)
        mag[3, 3] = 1.0
        loc = np.sort(np.random.rand(4))
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        dp4 = d.diff(4)
        expected = np.vstack([np.zeros((4, 4)), mag])
        assert dp4.delta_mags.shape == (8, 4)
        assert _ninf(np.array(dp4.delta_mags) - expected) == 0.0
