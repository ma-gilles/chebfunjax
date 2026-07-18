"""Port of MATLAB Chebfun tests/chebfun/test_permute.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_permute.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued chebfun now works, but permute is unrelated: MATLAB permute(f,[2 1])
# is the row/column transpose f.' and permute(f,[1 1])/[1 3] must error.
# chebfunjax has no permute() and no row-orientation transpose, so this stays
# skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax has no chebfun.permute()/row-column transpose (f.')"
)


class TestChebfunPermute:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
