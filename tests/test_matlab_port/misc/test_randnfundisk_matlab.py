"""Port of MATLAB Chebfun tests/misc/test_randnfundisk.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_randnfundisk.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test checks diskfun-valued output; chebfunjax returns grid samples (NOT YET PORTED assertion-for-assertion)")


class TestMiscRandnfundisk:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
