"""Port of MATLAB Chebfun tests/chebop2/test_generalVariableCoefficients.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_generalVariableCoefficients.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="All cases are variable-coefficient PDEs (m.*diff(u) with m a chebfun2); the scalar Chebop2 has no variable-coefficient path.")


class TestChebop2Generalvariablecoefficients:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
