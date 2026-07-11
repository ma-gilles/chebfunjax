"""Port of MATLAB Chebfun tests/chebop/test_cumsum.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop has no cumsum operator blocks")


class TestChebopCumsum:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
