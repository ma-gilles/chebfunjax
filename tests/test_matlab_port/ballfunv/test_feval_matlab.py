"""Port of MATLAB Chebfun tests/ballfunv/test_feval.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfunv/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="vector feval convention covered via components")


class TestBallfunvFeval:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
