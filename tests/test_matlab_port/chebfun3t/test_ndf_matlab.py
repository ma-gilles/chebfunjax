"""Port of MATLAB Chebfun tests/chebfun3t/test_ndf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3t/test_ndf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="ndf pins the full-tensor degrees of freedom: MATLAB chebfun3t.ndf == prod(size(f.coeffs)), the total size of the dense 3D coefficient tensor. chebfunjax's Chebfun3T is Tucker-backed (cols/rows/tubes factors + core) and exposes no full coefficient tensor or ndf method; its degrees of freedom are the rank-based Tucker count, a different quantity, so this full-tensor ndf assertion is not exercisable")


class TestChebfun3tNdf:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
