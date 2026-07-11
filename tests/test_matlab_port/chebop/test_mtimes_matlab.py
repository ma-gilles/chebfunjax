"""Port of MATLAB Chebfun tests/chebop/test_mtimes.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebop scalar*op composition not implemented")


class TestChebopMtimes:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
