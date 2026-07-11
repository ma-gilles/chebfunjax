"""Port of MATLAB Chebfun tests/chebop/test_domain.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_domain.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="constructor accepts domains leniently; MATLAB's error-identifier checks are MATLAB-specific")


class TestChebopDomain:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
