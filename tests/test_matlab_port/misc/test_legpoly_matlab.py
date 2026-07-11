"""Port of MATLAB Chebfun tests/misc/test_legpoly.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_legpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test builds a 201-column quasimatrix at degrees 900-1100; chebfunjax legpoly returns coefficient arrays (orthogonality is covered by the transforms ports)")


class TestMiscLegpoly:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
