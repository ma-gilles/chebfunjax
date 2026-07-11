"""Port of MATLAB Chebfun tests/chebfun/test_compose_chebfuns.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_compose_chebfuns.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no f(g) chebfun-of-chebfun composition")


class TestChebfunComposeChebfuns:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
