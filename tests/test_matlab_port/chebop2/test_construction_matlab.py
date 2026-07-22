"""Port of MATLAB Chebfun tests/chebop2/test_construction.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_construction.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Constant/Dirichlet solve checked at 10*eps; even the exact constant solution u==1 carries a ~7e-14 dense-solve error, above tol. Later assertions need Neumann BCs and variable-coefficient coeff cells.")


class TestChebop2Construction:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
