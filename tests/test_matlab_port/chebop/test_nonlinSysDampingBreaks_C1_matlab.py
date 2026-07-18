"""Port of MATLAB Chebfun tests/chebop/test_nonlinSysDampingBreaks_C1.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_nonlinSysDampingBreaks_C1.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="discretization-variant file (chebcolloc1/ultraS); chebfunjax has a single collocation discretization -- the piecewise-domain system itself is ported and passing in test_nonlinSysDampingBreaks_C2_matlab.py (FIXED, Fable 5 piecewise chebop)")


class TestChebopNonlinsysdampingbreaksC1:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
