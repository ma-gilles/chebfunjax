"""Port of MATLAB Chebfun tests/chebop/test_svds.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_svds.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="operator svds not implemented")


class TestChebopSvds:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
