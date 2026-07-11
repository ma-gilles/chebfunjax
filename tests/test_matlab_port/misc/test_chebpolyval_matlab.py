"""Port of MATLAB Chebfun tests/misc/test_chebpolyval.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_chebpolyval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no chebpolyval (quasimatrix of Chebyshev polys as chebfuns); coefficient transforms are tested in the transforms ports")


class TestMiscChebpolyval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
