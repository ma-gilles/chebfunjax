"""Port of MATLAB Chebfun tests/chebop/test_bcVectorInput.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_bcVectorInput.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="vector BC input not implemented")


class TestChebopBcvectorinput:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
