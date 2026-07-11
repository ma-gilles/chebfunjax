"""Port of MATLAB Chebfun tests/chebop/test_bcsyntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_bcsyntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="MATLAB bc string syntaxes ('dirichlet', 'neumann', @(x,u) ...) partially exist; string forms beyond periodic are not implemented")


class TestChebopBcsyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
