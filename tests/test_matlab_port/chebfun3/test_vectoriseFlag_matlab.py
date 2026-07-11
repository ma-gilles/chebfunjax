"""Port of MATLAB Chebfun tests/chebfun3/test_vectoriseFlag.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_vectoriseFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 constructor has no 'vectorize' flag")


class TestChebfun3Vectoriseflag:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
