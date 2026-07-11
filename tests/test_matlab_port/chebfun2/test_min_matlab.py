"""Port of MATLAB Chebfun tests/chebfun2/test_min.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_min.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2 has no min")


class TestChebfun2Min:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
