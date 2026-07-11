"""Port of MATLAB Chebfun tests/misc/test_lagpoly.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_lagpoly.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax has no lagpoly")


class TestMiscLagpoly:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
