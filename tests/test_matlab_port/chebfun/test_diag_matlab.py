"""Port of MATLAB Chebfun tests/chebfun/test_diag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued chebfun now works, but diag is unrelated to it: MATLAB diag(f)
# builds a multiplication operator D with D*g == f.*g.  chebfunjax has no such
# operator constructor, so this stays skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax has no chebfun.diag() (multiplication-operator constructor "
    "D with D*g == f.*g)"
)


class TestChebfunDiag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
