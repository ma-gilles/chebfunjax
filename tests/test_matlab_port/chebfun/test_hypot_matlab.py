"""Port of MATLAB Chebfun tests/chebfun/test_hypot.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_hypot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="chebfunjax Chebfun has no hypot; sqrt(f^2+g^2) covers the "
    "semantics and sqrt/power are ported")


class TestChebfunHypot:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
