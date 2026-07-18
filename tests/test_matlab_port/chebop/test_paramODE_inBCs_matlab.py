"""Port of MATLAB Chebfun tests/chebop/test_paramODE_inBCs.m (Fable 5).

Every case in this test places the unknown parameter in the boundary
conditions (and, where it appears in the operator, e.g. ``diff(x)-p``, it is
still determined only implicitly with no BC referencing it).  chebfunjax
solves parameter problems only when the parameter appears in the
differential operator AND is pinned by an endpoint boundary condition; the
BC-only / implicit cases leave the parameter at its initial value.  All
assertions therefore stay skipped with that precise reason.

Provenance
----------
MATLAB source : tests/chebop/test_paramODE_inBCs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="unknown parameters that appear only in the boundary conditions (or are "
    "determined only implicitly, with no BC referencing them) are not solved: the "
    "parameter block is a constant and the solver leaves it at its initial value. "
    "Parameter problems solve only when the parameter is in the operator AND pinned by "
    "an endpoint BC (see test_paramODE_matlab.py / test_paramODE_linearization) -- src gap"
)


class TestChebopParamodeInbcs:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
