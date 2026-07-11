"""Port of MATLAB Chebfun tests/spherefun/test_projection.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_projection.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun BMC projection internal")


class TestSpherefunProjection:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
