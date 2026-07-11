"""Port of MATLAB Chebfun tests/chebop/test_eigs_foxli.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_eigs_foxli.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Fox-Li integral operator not implemented")


class TestChebopEigsFoxli:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
