"""Port of MATLAB Chebfun tests/domain/test_poly.m (Fable 5).

Provenance
----------
MATLAB source : tests/domain/test_poly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB domain-class poly accessor has no counterpart")


class TestDomainPoly:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
