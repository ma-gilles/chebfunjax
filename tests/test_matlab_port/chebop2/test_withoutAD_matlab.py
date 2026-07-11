"""Port of MATLAB Chebfun tests/chebop2/test_withoutAD.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_withoutAD.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax Chebop2 solves scalar 2-D PDEs with lbc/rbc/ubc/dbc; MATLAB-specific syntaxes (coefficient chebfun2 inputs, generalized bc objects) absent -- basic Poisson/Helmholtz solves are golden-ref tested in tests/test_operators/test_chebop2_matlab.py")


class TestChebop2Withoutad:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
