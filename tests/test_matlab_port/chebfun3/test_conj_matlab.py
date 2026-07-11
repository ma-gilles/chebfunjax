"""Port of MATLAB Chebfun tests/chebfun3/test_conj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun3/test_conj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun3 has no conj()")


class TestChebfun3Conj:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
