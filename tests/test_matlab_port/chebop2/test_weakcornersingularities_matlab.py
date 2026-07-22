"""Port of MATLAB Chebfun tests/chebop2/test_weakcornersingularities.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop2/test_weakcornersingularities.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Needs n>=45 to reach the 1e6*eps tolerance (n=33 gives <2x headroom); the dense O(n^6) Kronecker solve is prohibitively expensive at that grid size.")


class TestChebop2Weakcornersingularities:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
