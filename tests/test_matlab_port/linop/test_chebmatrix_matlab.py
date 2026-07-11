"""Port of MATLAB Chebfun tests/linop/test_chebmatrix.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_chebmatrix.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Linop is an internal scalar-collocation solver (eigs/expm/matrix/solve) without MATLAB's chebmatrix-block algebra; the public operator surface is tested via the chebop ports; linop eigs/expm are exercised by chebop eigs_basic/expm ports")


class TestLinopChebmatrix:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
