"""Port of MATLAB Chebfun tests/chebfun/test_kronOp.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_kronOp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="kron(f, g', 'op') builds an OPERATOR (rank-1 operatorBlock f*(g'*.)) that is applied to and matrix-realized against a grid; cj.kron produces a Chebfun2 (tested in test_kron_matlab.py), and chebfunjax has no chebmatrix / operatorBlock matrix-realization (matrix(AC, n)) needed for the discrete-form assertions -- src gap")


class TestChebfunKronop:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
