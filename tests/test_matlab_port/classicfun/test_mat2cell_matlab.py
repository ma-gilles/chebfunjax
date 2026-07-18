"""Port of MATLAB Chebfun tests/classicfun/test_mat2cell.m (Fable 5).

Provenance
----------
MATLAB source : tests/classicfun/test_mat2cell.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued Bndfun now works, but there is no FUN-level mat2cell wrapper:
# the onefun (Chebtech2) exposes mat2cell, but Bndfun/Classicfun does not,
# so this stays skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax Bndfun/Classicfun has no mat2cell() wrapper (the onefun "
    "Chebtech2 has mat2cell, but the fun level does not expose it)"
)


class TestClassicfunMat2cell:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
