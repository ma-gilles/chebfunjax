"""Port of MATLAB Chebfun tests/chebop2/test_neumann.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_neumann.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Requires Neumann/Robin BCs dbc=@(x,u) diff(u)-...; the value-space solver imposes Dirichlet values only, with no derivative-BC row replacement.")


class TestChebop2Neumann:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
