"""Port of MATLAB Chebfun tests/chebfun3/test_min3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_min3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no min3")


class TestChebfun3Min3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
