"""Port of MATLAB Chebfun tests/spherefunv/test_empty.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_empty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="spherefunv: 'empty' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestSpherefunvEmpty:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
