"""Port of MATLAB Chebfun tests/chebfun/test_mrdivide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued chebfun / quasimatrices now work, but MATLAB f/A here is the
# matrix / least-squares mrdivide (mrdivide(A, B, 'ls')).  chebfunjax has no
# mrdivide operator on chebfun, so this stays skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax has no chebfun mrdivide (/, matrix / least-squares 'ls' solve)"
)


class TestChebfunMrdivide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
