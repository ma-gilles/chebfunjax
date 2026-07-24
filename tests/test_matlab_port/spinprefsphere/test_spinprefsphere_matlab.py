"""Port of MATLAB Chebfun tests/spinprefsphere/test_spinprefsphere.m (Fable 5).

Provenance
----------
MATLAB source : tests/spinprefsphere/test_spinprefsphere.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="chebfunjax spinsphere (operators.spinopsphere) takes kwargs and has no SpinPrefSphere preference object; the tested fields (Clim, dataplot, iterplot, Nplot, plot='movie', view) are plotting/movie preferences chebfunjax does not implement, and scheme selection (LIRK4 vs IMEXBDF4) is chosen internally from the linear part")


class TestSpinprefsphereSpinprefsphere:
    def test_all_matlab_assertions(self):
        raise NotImplementedError
