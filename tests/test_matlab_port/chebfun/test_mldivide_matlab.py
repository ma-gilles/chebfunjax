"""Port of MATLAB Chebfun tests/chebfun/test_mldivide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

# Array-valued chebfun / quasimatrices now work, but MATLAB A\B here is the
# least-squares backslash on chebfun quasimatrices (using chebpoly/legpoly
# quasimatrices, restrict, and 'ls' solves).  chebfunjax has no mldivide /
# backslash operator on chebfun, so this stays skipped on that precise gap.
pytestmark = pytest.mark.skip(
    reason="chebfunjax has no chebfun mldivide (\\, least-squares quasimatrix backslash)"
)


class TestChebfunMldivide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
