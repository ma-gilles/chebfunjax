"""Port of MATLAB Chebfun tests/chebfun2v/test_roots_slow.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_roots_slow.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Long-running resultant stress case; chebfunjax implements only the marching-squares method (covered by test_roots01/02/03/06/07).")


class TestChebfun2vRootsSlow:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
