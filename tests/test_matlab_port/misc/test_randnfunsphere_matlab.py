"""Port of MATLAB Chebfun tests/misc/test_randnfunsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfunsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks spherefun-valued output; chebfunjax returns grid samples (NOT YET PORTED assertion-for-assertion)")


class TestMiscRandnfunsphere:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
