"""Port of MATLAB Chebfun tests/chebfun3/test_guide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_guide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="guide script exercises chebfun3 composition (still absent); max3 and norm now exist")


class TestChebfun3Guide:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
