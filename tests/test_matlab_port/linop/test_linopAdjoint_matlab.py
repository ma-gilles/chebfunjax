"""Port of MATLAB Chebfun tests/linop/test_linopAdjoint.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_linopAdjoint.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="All 40 assertions call linopAdjoint(L, 'bvp'|'periodic'), which "
           "returns the adjoint linop together with the four side-condition "
           "descriptors (op, bcOpL, bcOpR, bcOpM); pass 25-40 need it for "
           "2x2 block systems ([D I; I D] and [D -D; I D]). chebfunjax has "
           "only a scalar-chebop adjoint (src/chebfunjax/operators/"
           "adjoint.py, covered by tests/test_operators/"
           "test_adjoint_matlab.py): it takes a Chebop rather than a linop, "
           "returns a Chebop rather than the 5-tuple, and does not handle "
           "systems. Porting linopAdjoint onto BlockLinop unblocks this file "
           "together with test_svds_matlab.py.")


class TestLinopLinopAdjoint:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
