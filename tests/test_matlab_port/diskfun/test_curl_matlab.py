"""Port of MATLAB Chebfun tests/diskfun/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scalar diskfun curl (stream-function) lives on Diskfunv; div/curl tested there")


class TestDiskfunCurl:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
