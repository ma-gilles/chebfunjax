"""Port of MATLAB Chebfun tests/chebop/test_intops.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_intops.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="integral operators not implemented")


class TestChebopIntops:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
