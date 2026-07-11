"""Port of MATLAB Chebfun tests/domain/test_polyfit.m (Fable 5).

Provenance
----------
MATLAB source : tests/domain/test_polyfit.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB domain-class polyfit; chebfun-level polyfit ported in chebfun")


class TestDomainPolyfit:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
