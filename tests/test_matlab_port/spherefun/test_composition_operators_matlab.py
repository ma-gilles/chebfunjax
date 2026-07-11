"""Port of MATLAB Chebfun tests/spherefun/test_composition_operators.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_composition_operators.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefun has no composition ops (cos(f), ...)")


class TestSpherefunCompositionOperators:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
