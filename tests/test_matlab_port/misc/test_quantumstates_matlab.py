"""Port of MATLAB Chebfun tests/misc/test_quantumstates.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_quantumstates.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks chebfun eigenstates; chebfunjax quantumstates covered by unit tests (NOT YET PORTED assertion-for-assertion)")


class TestMiscQuantumstates:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
