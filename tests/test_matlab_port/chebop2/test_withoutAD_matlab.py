"""Port of MATLAB Chebfun tests/chebop2/test_withoutAD.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_withoutAD.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Injects a manual low-rank operator via N.U/N.S/N.V (bypassing AD) and uses variable coefficients; chebfunjax has no low-rank operator-injection API.")


class TestChebop2Withoutad:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
