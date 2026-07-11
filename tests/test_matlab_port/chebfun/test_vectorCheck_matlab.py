"""Port of MATLAB Chebfun tests/chebfun/test_vectorCheck.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_vectorCheck.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB 'vectorize' flag does not exist")


class TestChebfunVectorcheck:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
