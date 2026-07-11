"""Port of MATLAB Chebfun tests/chebop/test_stringConstructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_stringConstructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="string operator constructor ('0.01*diff(u,2)+...') not implemented")


class TestChebopStringconstructor:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
