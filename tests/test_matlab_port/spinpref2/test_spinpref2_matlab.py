"""Port of MATLAB Chebfun tests/spinpref2/test_spinpref2.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinpref2/test_spinpref2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax spin2 has no preference object")


class TestSpinpref2Spinpref2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
