"""Port of MATLAB Chebfun tests/chebfun/test_fred.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_fred.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no fred (Fredholm integral operator on chebfun)")


class TestChebfunFred:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
