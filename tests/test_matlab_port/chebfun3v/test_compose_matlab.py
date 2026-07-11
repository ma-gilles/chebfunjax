"""Port of MATLAB Chebfun tests/chebfun3v/test_compose.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3v/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfun3v: 'compose' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestChebfun3vCompose:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
