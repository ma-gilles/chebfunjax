"""Port of MATLAB Chebfun tests/chebfun2v/test_roots09.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots09.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Cross-checks ms vs resultant (resultant not implemented in chebfunjax). Correctness covered by test_roots01/02/03/06/07.")


class TestChebfun2vRoots09:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
