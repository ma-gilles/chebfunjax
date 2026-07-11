"""Port of MATLAB Chebfun tests/chebop/test_exactInitial.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_exactInitial.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="N.init exact-solution shortcut semantics not implemented")


class TestChebopExactinitial:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
