"""Port of MATLAB Chebfun tests/chebtech/test_iszero.m (Opus 4.8).

chebfunjax Chebtech has NO ``iszero()`` method, so every assertion in this
file is skipped with a precise reason.  (The MATLAB test also relies on
directly assigning 2-D array-valued ``f.coeffs``, which chebfunjax scalar
techs do not support.)  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no iszero() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechIszero:
    def test_iszero_columns_mixed(self, Tech):
        # pass(n,1): iszero(f) == [1 0 0] for coeffs [0 1 0; 0 0 NaN]
        pytest.skip(_REASON)

    def test_iszero_row_mixed(self, Tech):
        # pass(n,2): iszero(f) == [1 0 0] for coeffs [0 NaN 1]
        pytest.skip(_REASON)

    def test_iszero_column_mixed(self, Tech):
        # pass(n,3): iszero(f) == 0 for coeffs [0 NaN 1]'
        pytest.skip(_REASON)

    def test_iszero_all_zero(self, Tech):
        # pass(n,4): iszero(f) == 1 for coeffs zeros(3,1)
        pytest.skip(_REASON)

    def test_iszero_nan(self, Tech):
        # pass(n,5): iszero(f) == 0 for coeffs NaN
        pytest.skip(_REASON)
