"""Port of MATLAB Chebfun tests/spinpref3/test_spinpref3.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinpref3/test_spinpref3.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax spin3 has no preference object")


class TestSpinpref3Spinpref3:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
