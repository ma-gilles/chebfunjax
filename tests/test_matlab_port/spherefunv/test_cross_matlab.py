"""Port of MATLAB Chebfun tests/spherefunv/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefunv cross needs a 3-Cartesian-component representation (MATLAB builds 3-component fields, e.g. the unit normal); chebfunjax Spherefunv is 2-component. Out of scope (representation overhaul).")


class TestSpherefunvCross:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
