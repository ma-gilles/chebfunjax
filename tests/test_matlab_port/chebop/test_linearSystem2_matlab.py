"""Port of MATLAB Chebfun tests/chebop/test_linearSystem2.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_linearSystem2.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB u{1}/u{2} cell-indexing NOTATION; the same class of linear systems is ported in test_linearSystem1_matlab.py (multi-argument form); the cell syntax itself is MATLAB-specific")


class TestChebopLinearsystem2:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
