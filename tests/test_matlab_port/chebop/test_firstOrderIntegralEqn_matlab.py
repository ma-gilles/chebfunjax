"""Port of MATLAB Chebfun tests/chebop/test_firstOrderIntegralEqn.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_firstOrderIntegralEqn.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="integral-equation operators (fred/volt) not implemented")


class TestChebopFirstorderintegraleqn:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
