"""Port of MATLAB Chebfun tests/chebfun/test_constructor_inputs_periodic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_inputs_periodic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="periodic ctor input variants do not exist")


class TestChebfunConstructorInputsPeriodic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
