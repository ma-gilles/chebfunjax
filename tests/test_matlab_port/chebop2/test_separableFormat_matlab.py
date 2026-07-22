"""Port of MATLAB Chebfun tests/chebop2/test_separableFormat.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_separableFormat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Needs the chebop2.separableFormat low-rank-of-PDO API returning {U,S,V} cells, plus variable-coefficient PDOs; neither exists.")


class TestChebop2Separableformat:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
