"""Port of MATLAB Chebfun tests/deltafun/test_minandmax.m (Fable 5).

``minandmax`` takes the funPart extrema and then lets a positive zeroth-order
delta drive the maximum to +Inf (at the first positive-delta location) and a
negative delta drive the minimum to -Inf.  Higher-order deltas are ignored.
The empty-Deltafun case (pass 1) is skipped: chebfunjax has no empty Deltafun
representation.

Provenance
----------
MATLAB source : tests/deltafun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

TOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))


def _f():
    return Bndfun.from_function(jnp.exp, DOM)


class TestDeltafunMinandmax:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty Deltafun representation")

    # --- all positive zeroth-order deltas ---
    def _pos(self):
        mag = np.vstack([[1.0, 2.0, 3.0, 4.0, 5.0], np.random.randn(5)])
        loc = np.sort(np.random.rand(5))
        return Deltafun(_f(), jnp.asarray(loc), jnp.asarray(mag)), loc

    def test_pos_min_is_funpart_min(self):
        # pass(2): vals(1) ~ exp(-1)
        d, _ = self._pos()
        vals, _ = d.minandmax()
        assert abs(float(vals[0]) - np.exp(-1)) < TOL

    def test_pos_max_is_inf(self):
        # pass(3): isinf(vals(2)) && vals(2) > 0
        d, _ = self._pos()
        vals, _ = d.minandmax()
        assert np.isinf(float(vals[1])) and float(vals[1]) > 0

    def test_pos_min_pos(self):
        # pass(4): pos(1) ~ -1
        d, _ = self._pos()
        _, pos = d.minandmax()
        assert abs(float(pos[0]) - (-1.0)) < TOL

    def test_pos_max_pos(self):
        # pass(5): pos(2) == loc(1)
        d, loc = self._pos()
        _, pos = d.minandmax()
        assert abs(float(pos[1]) - loc[0]) < TOL

    # --- all negative zeroth-order deltas ---
    def _neg(self):
        mag = np.vstack([[-1.0, -2.0, -3.0, -4.0, -5.0], np.random.rand(5)])
        loc = np.sort(np.random.rand(5))
        return Deltafun(_f(), jnp.asarray(loc), jnp.asarray(mag)), loc

    def test_neg_min_is_minus_inf(self):
        # pass(6): isinf(vals(1)) && vals(1) < 0
        d, _ = self._neg()
        vals, _ = d.minandmax()
        assert np.isinf(float(vals[0])) and float(vals[0]) < 0

    def test_neg_max_is_funpart_max(self):
        # pass(7): vals(2) ~ exp(1)
        d, _ = self._neg()
        vals, _ = d.minandmax()
        assert abs(float(vals[1]) - np.exp(1)) < TOL

    def test_neg_min_pos(self):
        # pass(8): pos(1) == loc(1)
        d, loc = self._neg()
        _, pos = d.minandmax()
        assert abs(float(pos[0]) - loc[0]) < TOL

    def test_neg_max_pos(self):
        # pass(9): pos(2) ~ 1
        d, _ = self._neg()
        _, pos = d.minandmax()
        assert abs(float(pos[1]) - 1.0) < TOL

    # --- mixed signs ---
    def _mixed(self):
        mag = np.vstack([[-1.0, 2.0, -3.0, -4.0, 5.0], np.random.rand(5)])
        loc = np.sort(np.random.rand(5))
        return Deltafun(_f(), jnp.asarray(loc), jnp.asarray(mag)), loc

    def test_mixed_min_is_minus_inf(self):
        # pass(10): isinf(vals(1)) && vals(1) < 0
        d, _ = self._mixed()
        vals, _ = d.minandmax()
        assert np.isinf(float(vals[0])) and float(vals[0]) < 0

    def test_mixed_max_is_inf(self):
        # pass(11): isinf(vals(2)) && vals(2) > 0
        d, _ = self._mixed()
        vals, _ = d.minandmax()
        assert np.isinf(float(vals[1])) and float(vals[1]) > 0

    def test_mixed_min_pos(self):
        # pass(12): pos(1) == loc(1) (first negative delta at index 0)
        d, loc = self._mixed()
        _, pos = d.minandmax()
        assert abs(float(pos[0]) - loc[0]) < TOL

    def test_mixed_max_pos(self):
        # pass(13): pos(2) == loc(2) (first positive delta at index 1)
        d, loc = self._mixed()
        _, pos = d.minandmax()
        assert abs(float(pos[1]) - loc[1]) < TOL
