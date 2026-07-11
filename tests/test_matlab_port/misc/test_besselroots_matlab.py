"""Port of MATLAB Chebfun tests/misc/test_besselroots.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_besselroots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from chebfunjax.utils.specfun import besselroots


def _res(v):
    r = np.asarray(besselroots(v, 30))
    return float(np.max(np.abs(jv(v, r))))


class TestBesselroots:
    def test_order_zero(self):
        assert _res(0) < 1e-12

    def test_order_minus_one(self):
        assert _res(-1) < 1e-10

    def test_order_2p5(self):
        assert _res(2.5) < 1e-10

    def test_order_4(self):
        assert _res(4) < 1e-10

    def test_order_6_first_roots_looser(self):
        r = np.asarray(besselroots(6, 30))
        res = np.abs(jv(6, r))
        assert float(np.max(res[:6])) < 1e-2
        assert float(np.max(res[6:])) < 1e-8

    def test_order_7_first_roots_looser(self):
        r = np.asarray(besselroots(7, 30))
        res = np.abs(jv(7, r))
        assert float(np.max(res[:6])) < 1e-2
        assert float(np.max(res[6:])) < 1e-8
