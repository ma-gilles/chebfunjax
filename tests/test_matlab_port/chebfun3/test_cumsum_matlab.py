"""Port of MATLAB Chebfun tests/chebfun3/test_cumsum.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_cumsum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no cumsum")


class TestChebfun3Cumsum:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
