"""Port of MATLAB Chebfun tests/misc/test_trigBary.m (Fable 5).

FIXED: trigBary added in the Fable 5 audit (Henrici/Berrut 2nd-form
trigonometric barycentric interpolation with arbitrary-node weights).

Provenance
----------
MATLAB source : tests/misc/test_trigBary.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

import chebfunjax as cj
from chebfunjax.utils.quadrature import trigpts

TOL = 1.0e-12
RNG = np.random.default_rng(3453)
XR = 2 * RNG.random(1000) - 1


def _p1(x):
    return (np.cos(4 * np.pi * x) - 2 * np.sin(3 * np.pi * x)
            + 3 * np.sin(2 * np.pi * x) - 2 * np.cos(np.pi * x) + 1)


class TestTrigBary:
    def test_trig_polynomial(self):
        # pass(1)-(2): interpolation of a trig polynomial from trigpts
        for n in (10, 1001):
            xk = np.asarray(trigpts(n, (-1, 1))[0])
            y = cj.trigBary(XR, _p1(xk), xk, (-1, 1))
            assert np.max(np.abs(y - _p1(XR))) < TOL, n

    def test_interpolation_at_nodes(self):
        # pass(3)
        xk = np.asarray(trigpts(10, (-1, 1))[0])
        y = cj.trigBary(xk, _p1(xk), xk, (-1, 1))
        assert np.max(np.abs(y - _p1(xk))) < TOL

    def test_nonequispaced_nodes(self):
        # pass(4): first-kind Chebyshev points as nodes
        xk = -np.cos((2 * np.arange(1, 9) - 1) * np.pi / 16)

        def p2(x):
            return (2 * np.sin(3 * np.pi * x)
                    + 3 * np.sin(2 * np.pi * x)
                    - 2 * np.cos(np.pi * x) + 1)

        y = cj.trigBary(XR, p2(xk), xk, (-1, 1))
        assert np.max(np.abs(y - p2(XR))) < TOL

    def test_array_valued(self):
        # pass(5)-(6): two columns, domain [-2, 2]
        def q(x):
            return np.stack(
                [0.45 + np.sin(np.pi * x),
                 0.32 + np.sin(np.pi * x) + np.cos(2 * np.pi * x)],
                axis=-1)

        rng = np.random.default_rng(7)
        xk = np.sort(-2 + 4 * rng.random(9))
        xr = -2 + 4 * rng.random(100)
        y = cj.trigBary(xr, q(xk), xk, (-2, 2))
        assert np.max(np.abs(y - q(xr))) < 1e-10
