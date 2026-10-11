"""Native slice checks with the effective disk preference and scale controls.

Provenance: @diskfun/constructor.m sets pref.chebfuneps=getTol before PhaseTwo;
@chebtech/happinessCheck.m and @trigtech/happinessCheck.m, pin7574c77.
Power-of-two amplitude cases are checker controls, not whole constructors.
"""

from pathlib import Path

import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import _is_happy_cheb, _is_happy_trig

FIXTURE = Path(__file__).parents[1] / "references/disk_phase2_happiness_native.npz"


@pytest.mark.parametrize("case", range(9))
def test_native_effective_preference_slice_happiness(case):
    with np.load(FIXTURE) as data:
        cols = data[f"cols_{case}"]
        rows = data[f"rows_{case}"]
        expected = data[f"expected_{case}"]
        tol = float(data[f"tol_{case}"])
        vscale = float(data[f"vscale_{case}"])
    actual = [
        _is_happy_cheb(cols, tol, vscale),
        _is_happy_trig(rows, tol, vscale),
    ]
    np.testing.assert_array_equal(actual, expected)
