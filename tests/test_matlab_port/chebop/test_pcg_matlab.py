"""Port of MATLAB Chebfun tests/chebop/test_pcg.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_pcg.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="operator pcg not implemented")


class TestChebopPcg:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
