"""Port of MATLAB Chebfun tests/unbndfun/test_createMap.m (Opus 4.8).

``unbndfun.createMap(dom)`` builds the nonlinear algebraic map between the
reference interval [-1, 1] and the unbounded physical domain.  chebfunjax
implements the same three maps as pure module-level helpers in
``chebfunjax.fun.unbndfun`` (forward and inverse for right-inf, left-inf and
both-inf).  We test the forward map at +-1 and the inverse map at +-1e100,
exactly as MATLAB checks ``map.For`` and ``map.Inv``.

Provenance
----------
MATLAB source : tests/unbndfun/test_createMap.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.fun.unbndfun import (
    _forward_both,
    _forward_left,
    _forward_right,
    _inverse_both,
    _inverse_left,
    _inverse_right,
)

EPS = float(np.finfo(np.float64).eps)
TOL = 1e1 * EPS
INF = np.inf


class TestUnbndfunCreateMap:
    def test_doubly_unbounded(self):
        # dom = [-Inf Inf]
        forw = np.array([_forward_both(np.float64(-1.0)),
                         _forward_both(np.float64(1.0))])
        assert np.all(forw == np.array([-INF, INF]))
        inv = np.array([_inverse_both(np.float64(-1e100)),
                        _inverse_both(np.float64(1e100))])
        assert np.all(np.abs(inv - np.array([-1.0, 1.0])) < TOL)

    def test_left_unbounded(self):
        # dom = [-Inf 3]
        b = 3.0
        forw = np.array([_forward_left(np.float64(-1.0), b),
                         _forward_left(np.float64(1.0), b)])
        assert np.all(forw == np.array([-INF, b]))
        inv = np.array([_inverse_left(np.float64(-1e100), b),
                        _inverse_left(np.float64(3.0), b)])
        assert np.all(inv == np.array([-1.0, 1.0]))

    def test_right_unbounded(self):
        # dom = [-100 Inf]
        a = -100.0
        forw = np.array([_forward_right(np.float64(-1.0), a),
                         _forward_right(np.float64(1.0), a)])
        assert np.all(forw == np.array([a, INF]))
        inv = np.array([_inverse_right(np.float64(-100.0), a),
                        _inverse_right(np.float64(1e100), a)])
        assert np.all(inv == np.array([-1.0, 1.0]))
