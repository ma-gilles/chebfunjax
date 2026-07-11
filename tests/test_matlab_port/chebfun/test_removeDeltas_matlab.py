"""Port of MATLAB Chebfun tests/chebfun/test_removeDeltas.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_removeDeltas.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no removeDeltas (deltas field is static metadata)")


class TestChebfunRemovedeltas:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
