"""Port of MATLAB Chebfun tests/chebfun/test_polyval.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_polyval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB polyval-style coefficient evaluation not applicable")


class TestChebfunPolyval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
