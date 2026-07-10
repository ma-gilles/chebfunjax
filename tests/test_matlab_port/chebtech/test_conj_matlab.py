"""Port of MATLAB Chebfun tests/chebtech/test_conj.m (Opus 4.8).

chebfunjax Chebtech has NO ``conj()`` method, so every assertion in this
file is skipped with a precise reason.  No assertion is silently dropped.

Provenance
----------
MATLAB source : tests/chebtech/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

_REASON = "chebfunjax Chebtech has no conj() method"


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechConj:
    def test_conj_scalar(self, Tech):
        # pass(n,1): conj(cos(x) + 1i*sin(x)) == cos(x) - 1i*sin(x)
        pytest.skip(_REASON)

    def test_conj_array(self, Tech):
        # pass(n,2): conj([cos+1i*sin, -exp(1i*x)]) columns conjugated
        pytest.skip(_REASON)
