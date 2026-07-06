"""Port of MATLAB Chebfun tests/deltafun/test_minandmax.m (Opus 4.8).

chebfunjax's Deltafun has no (delta-aware) ``minandmax`` method: a Deltafun with
a positive delta has +Inf as its max (at the delta location) and with a negative
delta has -Inf as its min.  This behaviour is not implemented, so every
assertion is skipped with a precise reason.

Provenance
----------
MATLAB source : tests/deltafun/test_minandmax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Deltafun has no (delta-aware) minandmax() method"
)


class TestDeltafunMinandmax:
    def test_empty(self):
        # pass(1): [vals,pos]=minandmax(deltafun()) both empty
        pass

    def test_pos_delta_min_at_endpoint(self):
        # pass(2): vals(1) ~ exp(-1)
        pass

    def test_pos_delta_max_is_plus_inf(self):
        # pass(3): isinf(vals(2)) && vals(2) > 0
        pass

    def test_pos_delta_min_position(self):
        # pass(4): pos(1) ~ -1
        pass

    def test_pos_delta_max_position(self):
        # pass(5): pos(2) ~ loc(1)
        pass

    def test_neg_delta_min_is_minus_inf(self):
        # pass(6): isinf(vals(1)) && vals(1) < 0
        pass

    def test_neg_delta_max_at_endpoint(self):
        # pass(7): vals(2) ~ exp(1)
        pass

    def test_neg_delta_min_position(self):
        # pass(8): pos(1) ~ loc(1)
        pass

    def test_neg_delta_max_position(self):
        # pass(9): pos(2) ~ 1
        pass

    def test_mixed_delta_min_is_minus_inf(self):
        # pass(10): isinf(vals(1)) && vals(1) < 0
        pass

    def test_mixed_delta_max_is_plus_inf(self):
        # pass(11): isinf(vals(2)) && vals(2) > 0
        pass

    def test_mixed_delta_min_position(self):
        # pass(12): pos(1) ~ loc(1)
        pass

    def test_mixed_delta_max_position(self):
        # pass(13): pos(2) ~ loc(2)
        pass
