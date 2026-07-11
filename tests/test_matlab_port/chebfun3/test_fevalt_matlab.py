"""Port of MATLAB Chebfun tests/chebfun3/test_fevalt.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_fevalt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no fevalt (tensor evaluation) method")


class TestChebfun3Fevalt:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
