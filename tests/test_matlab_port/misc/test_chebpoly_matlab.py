"""Port of MATLAB Chebfun tests/misc/test_chebpoly.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_chebpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB test builds chebfun quasimatrices; chebfunjax chebpoly returns coefficient arrays")


class TestMiscChebpoly:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
