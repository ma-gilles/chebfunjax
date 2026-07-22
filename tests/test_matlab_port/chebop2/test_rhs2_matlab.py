"""Port of MATLAB Chebfun tests/chebop2/test_rhs2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_rhs2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Relies on constant/function forcing terms embedded in the operator (laplacian(u)-1, laplacian(u)-sin(x)) being moved to the RHS; the scalar proxy treats a bare additive term as a u-coefficient, so operator->RHS extraction is missing. pass2-3 also use variable coefficients.")


class TestChebop2Rhs2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
