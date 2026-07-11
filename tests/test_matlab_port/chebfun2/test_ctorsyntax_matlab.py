"""Port of MATLAB Chebfun tests/chebfun2/test_ctorsyntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_ctorsyntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="string/vectorize/coeffs ctor syntaxes do not exist in chebfunjax")


class TestChebfun2Ctorsyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
