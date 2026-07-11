"""Port of MATLAB Chebfun tests/chebfun3/test_chebcoeffs3.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_chebcoeffs3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no chebcoeffs3 accessor")


class TestChebfun3Chebcoeffs3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
