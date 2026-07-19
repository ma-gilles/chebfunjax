"""Port of MATLAB Chebfun tests/chebfun2v/test_roots_syntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots_syntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Checks the ms/resultant/auto method-selection syntaxes agree; chebfunjax exposes a single common-zero method, so there is nothing to cross-check. roots(f,g) and Chebfun2v.roots are exercised in test_roots01/02/03/06/07.")


class TestChebfun2vRootsSyntax:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
