"""Port of MATLAB Chebfun tests/adchebfun/test_innerProduct.m (Fable 5).

Provenance
----------
MATLAB source : tests/adchebfun/test_innerProduct.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax uses JAX automatic differentiation instead of the adchebfun operator-overloading AD class (user decision: JAX AD is the direct counterpart); chebop Newton linearization is exercised by the chebop ports")


class TestAdchebfunInnerproduct:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
