"""Port of MATLAB Chebfun tests/chebfun/test_idlt.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_idlt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no idlt")


class TestChebfunIdlt:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
