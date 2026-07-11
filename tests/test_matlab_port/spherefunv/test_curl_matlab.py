"""Port of MATLAB Chebfun tests/spherefunv/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="spherefunv: 'curl' targets a missing feature (MATLAB accessor/op not implemented in chebfunjax)")


class TestSpherefunvCurl:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
