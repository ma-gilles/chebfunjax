"""Port of MATLAB Chebfun tests/deltafun/test_restrict.m (Fable 5).

``restrict`` splits the funPart and distributes delta functions to the
subinterval(s) that contain them, halving any delta that sits exactly on an
interior breakpoint so the two adjacent pieces each carry half.  A piece with
no deltas is returned as a bare Bndfun.  The empty-Deltafun case (pass 1) is
skipped: chebfunjax has no empty Deltafun representation.

Provenance
----------
MATLAB source : tests/deltafun/test_restrict.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DOM = Domain((-1.0, 1.0))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestDeltafunRestrict:
    def test_restrict_empty(self):
        # pass(1): isempty(restrict(deltafun(), [-.5, .5]))
        assert Deltafun.empty().restrict([-0.5, 0.5]).isempty()

    def test_restrict_away_from_delta_left(self):
        # pass(2): ~isa(restrict(d, [-1, -.5]), 'deltafun') (delta at 0 excluded)
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        assert not isinstance(d.restrict([-1.0, -0.5]), Deltafun)

    def test_restrict_away_from_delta_right(self):
        # pass(3): ~isa(restrict(d, [.5, 1]), 'deltafun')
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        assert not isinstance(d.restrict([0.5, 1.0]), Deltafun)

    def test_restrict_containing_delta(self):
        # pass(4): anyDelta(restrict(d, [-.5, .5]))
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        g = d.restrict([-0.5, 0.5])
        assert isinstance(g, Deltafun) and g.has_deltas

    def _multi_d(self):
        f = Bndfun.from_function(jnp.exp, DOM)
        return Deltafun(f, jnp.array([-0.5, -0.25, 0.0, 1.0]),
                        jnp.array([1.0, 1.0, 1.0, 1.0]))

    def test_restrict_multi_locations(self):
        # pass(5): d1.deltaLoc == [-.5, -.25, 0]
        A = self._multi_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d1 = A[0]
        assert _ninf(d1.delta_locs - np.array([-0.5, -0.25, 0.0])) == 0.0

    def test_restrict_multi_middle_domain(self):
        # pass(6): d2.funPart.domain == [0, .5]
        A = self._multi_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d2 = A[1]
        dom = d2.funPart.domain if isinstance(d2, Deltafun) else d2.domain
        assert float(dom.a) == 0.0 and float(dom.b) == 0.5

    def test_restrict_multi_last_mag(self):
        # pass(7): d3.deltaMag == 1 (delta at right endpoint, not halved)
        A = self._multi_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d3 = A[2]
        assert _ninf(d3.delta_mags - np.array([[1.0]])) == 0.0

    def _grid_d(self):
        f = Bndfun.from_function(jnp.exp, DOM)
        loc = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
        mag = np.ones(7)
        return Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))

    def test_restrict_grid_piece1(self):
        # pass(8): d1.deltaLoc == [-1 -.5 -.25 0], d1.deltaMag == [1 1 1 .5]
        A = self._grid_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d1 = A[0]
        assert _ninf(d1.delta_locs - np.array([-1.0, -0.5, -0.25, 0.0])) == 0.0
        assert _ninf(d1.delta_mags - np.array([[1.0, 1.0, 1.0, 0.5]])) == 0.0

    def test_restrict_grid_piece2(self):
        # pass(9): d2.deltaLoc == [0 .25 .5], d2.deltaMag == [.5 1 .5]
        A = self._grid_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d2 = A[1]
        assert _ninf(d2.delta_locs - np.array([0.0, 0.25, 0.5])) == 0.0
        assert _ninf(d2.delta_mags - np.array([[0.5, 1.0, 0.5]])) == 0.0

    def test_restrict_grid_piece3(self):
        # pass(10): d3.deltaLoc == [.5 1], d3.deltaMag == [.5 1]
        A = self._grid_d().restrict([-1.0, 0.0, 0.5, 1.0])
        d3 = A[2]
        assert _ninf(d3.delta_locs - np.array([0.5, 1.0])) == 0.0
        assert _ninf(d3.delta_mags - np.array([[0.5, 1.0]])) == 0.0
