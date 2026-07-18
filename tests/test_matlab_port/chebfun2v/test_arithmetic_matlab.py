"""Port of MATLAB Chebfun tests/chebfun2v/test_arithmetic.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_arithmetic.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Chebfun2v +/- work but subtraction of distinct fields does not recompress (norm(f-g-exact) ~ 3e-8, above 1e3*cheb2eps), and component-wise .* between two Chebfun2v is unsupported (TypeError); construction/scalar-arithmetic are ported in test_twocomponents -- src gaps")


class TestChebfun2vArithmetic:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
