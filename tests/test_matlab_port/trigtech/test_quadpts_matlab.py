"""Port of MATLAB Chebfun tests/trigtech/test_quadpts.m (Opus 4.8).

Tests the trigonometric quadrature weights ``trigtech.quadwts(n)``.
chebfunjax's trigtech exposes definite integration through ``sum`` (== 2*c_0)
but does not expose a standalone ``quadwts`` weight vector, so each weight
assertion is skipped with that precise reason.

Provenance
----------
MATLAB source : tests/trigtech/test_quadpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest


class TestTrigtechQuadpts:
    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method (integration via sum only)")
    def test_weights_sum_to_two(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_annihilate_sin(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_annihilate_sin_cos(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_integrate_sin_squared(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_integrate_cos_squared(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_empty(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_n_one(self):
        pass

    @pytest.mark.skip(reason="chebfunjax trigtech has no quadwts(n) method")
    def test_weights_n_two(self):
        pass
