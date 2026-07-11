"""Port of MATLAB Chebfun tests/chebfun3/test_isreal.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_isreal.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no isreal")


class TestChebfun3Isreal:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
