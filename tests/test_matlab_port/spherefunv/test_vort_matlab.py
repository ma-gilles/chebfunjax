"""Port of MATLAB Chebfun tests/spherefunv/test_vort.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_vort.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="Spherefunv vorticity needs a 3-Cartesian-component representation + surface differential operators; chebfunjax Spherefunv is 2-component. Out of scope (representation overhaul).")


class TestSpherefunvVort:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
