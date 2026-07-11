"""Port of MATLAB Chebfun tests/chebop/test_eigs_piecewise.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_eigs_piecewise.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="piecewise-coefficient eigs not implemented")


class TestChebopEigsPiecewise:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
