"""Port of MATLAB Chebfun tests/chebop2/test_eulerTricomi.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_eulerTricomi.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Euler-Tricomi PDE has variable coefficients (x.*diff(u,2,1)); not representable in the scalar constant-coefficient Chebop2.")


class TestChebop2Eulertricomi:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
