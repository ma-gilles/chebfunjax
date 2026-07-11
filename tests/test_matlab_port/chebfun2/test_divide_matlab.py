"""Port of MATLAB Chebfun tests/chebfun2/test_divide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_divide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="requires composition/abs ops beyond scalar rdivide (hand-ported cases live in test_rdivide)")


class TestChebfun2Divide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
