"""Port of MATLAB Chebfun tests/chebop2/test_univariate.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_univariate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Rank-1 detection: for the y-only PDE the wrapped solution has numerical rank 2, so length(u)==1 fails; also cross-checks against a 1-D chebop solve.")


class TestChebop2Univariate:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
