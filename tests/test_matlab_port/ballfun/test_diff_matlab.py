"""Port of MATLAB Chebfun tests/ballfun/test_diff.m (Fable 5).

Adversarial Cartesian-derivative sweep (the check that exposed the
diskfun calculus bugs; ballfun passes all classes).

Provenance
----------
MATLAB source : tests/ballfun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import pytest

from chebfunjax.ballfun.ballfun import Ballfun

from ._helpers import X0, Y0, Z0, val

CASES = [
    (lambda x, y, z: x, (1.0, 0.0, 0.0)),
    (lambda x, y, z: y, (0.0, 1.0, 0.0)),
    (lambda x, y, z: z, (0.0, 0.0, 1.0)),
    (lambda x, y, z: x * x, (2 * X0, 0.0, 0.0)),
    (lambda x, y, z: x * y, (Y0, X0, 0.0)),
    (lambda x, y, z: z * z, (0.0, 0.0, 2 * Z0)),
    (lambda x, y, z: x * y * z, (Y0 * Z0, X0 * Z0, X0 * Y0)),
]


class TestBallfunDiff:
    @pytest.mark.parametrize("i", range(len(CASES)))
    def test_cartesian_partials(self, i):
        op, want = CASES[i]
        f = Ballfun.from_function(op)
        for dim in (1, 2, 3):
            got = val(f.diff(dim))
            assert abs(got - want[dim - 1]) < 1e-7, (i, dim)
