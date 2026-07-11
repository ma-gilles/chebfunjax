"""Port of MATLAB Chebfun tests/chebfun2/test_vectoriseFlag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_vectoriseFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 constructor has no 'vectorize' flag")


class TestChebfun2Vectoriseflag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
