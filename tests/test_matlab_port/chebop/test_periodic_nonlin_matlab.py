"""Port of MATLAB Chebfun tests/chebop/test_periodic_nonlin.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_periodic_nonlin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="nonlinear periodic solve not implemented (linear periodic is)")


class TestChebopPeriodicNonlin:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
