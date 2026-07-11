"""Port of MATLAB Chebfun tests/chebfun3/test_max3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_max3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no max3")


class TestChebfun3Max3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
