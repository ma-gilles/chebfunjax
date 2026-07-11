"""Port of MATLAB Chebfun tests/spinpref/test_spinpref.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinpref/test_spinpref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax spin has no preference object (kwargs instead)")


class TestSpinprefSpinpref:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
