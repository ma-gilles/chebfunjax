"""Port of MATLAB Chebfun tests/chebfun3/test_domainvolume.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_domainvolume.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no domainvolume")


class TestChebfun3Domainvolume:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
