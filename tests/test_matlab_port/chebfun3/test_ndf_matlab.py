"""Port of MATLAB Chebfun tests/chebfun3/test_ndf.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_ndf.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no ndf (number of degrees of freedom)")


class TestChebfun3Ndf:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
