"""Port of MATLAB Chebfun tests/spherefun/test_curl.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_curl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="scalar Spherefun has no curl (vorticity of a scalar stream fn); Spherefunv tested separately")


class TestSpherefunCurl:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
