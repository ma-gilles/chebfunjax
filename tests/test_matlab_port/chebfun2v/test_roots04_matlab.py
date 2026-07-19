"""Port of MATLAB Chebfun tests/chebfun2v/test_roots04.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots04.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Cross-checks the ms and resultant methods; chebfunjax has only marching-squares. The 2nd case is a degree-18 resultant stress test. Common-zero correctness is covered by test_roots01/02/03/06/07.")


class TestChebfun2vRoots04:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
