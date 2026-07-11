"""Port of MATLAB Chebfun tests/trigspec/test_multmat.m (Fable 5).

Provenance
----------
MATLAB source : tests/trigspec/test_multmat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no trigspec discretization class; periodic solves use Fourier collocation tested in the chebop periodic port")


class TestTrigspecMultmat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
