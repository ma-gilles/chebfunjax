"""Port of MATLAB Chebfun tests/chebop/test_autoVectorize.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_autoVectorize.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="no vectorize flag")


class TestChebopAutovectorize:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
