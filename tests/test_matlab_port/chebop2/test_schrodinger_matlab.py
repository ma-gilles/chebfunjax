"""Port of MATLAB Chebfun tests/chebop2/test_schrodinger.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_schrodinger.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Schrodinger operator has complex coefficients (1i*diff(u,1,1)); chebfunjax Chebop2 assembles a real float64 Kronecker system.")


class TestChebop2Schrodinger:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
