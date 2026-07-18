"""Port of MATLAB Chebfun tests/chebfun/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued chebfun now works, but vertcat is unrelated: MATLAB [f; g] builds
# a chebmatrix (a 2-D block matrix with a `.blocks` cell array).  chebfunjax has
# no chebmatrix type, so this stays skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax has no chebmatrix type ([f; g] vertical concatenation of "
    "chebfuns into a 2-D block matrix)"
)


class TestChebfunVertcat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
