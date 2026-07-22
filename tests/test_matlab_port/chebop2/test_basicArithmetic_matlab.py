"""Port of MATLAB Chebfun tests/chebop2/test_basicArithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_basicArithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Uses variable-coefficient PDOs (x.*diff(u,2,1)); chebfunjax Chebop2 stores only scalar constant coefficients, so the coeff-cell comparison has no analogue.")


class TestChebop2Basicarithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
