"""Port of MATLAB Chebfun tests/chebtech1/test_chebpts.m (Opus 4.8).

Self-validating: 1st-kind Chebyshev points, quadrature weights and
barycentric weights are checked against closed-form exacts at the SAME
tolerance MATLAB uses (10*eps).

MATLAB's ``chebtech1.chebpts(n)`` returns ``[x, w, v]``.  In chebfunjax
these are three separate helpers:
    x = chebpts(n, kind=1)
    w = chebweights(n, kind=1)          (Fejér-1, matches MATLAB)
    v = _cheb1_barywts(n)               (1st-kind barywts, matches MATLAB)
chebfunjax returns 1-D arrays of shape ``(n,)`` throughout (no MATLAB
column/row distinction), so size checks are adapted to ``.shape == (n,)``.

All assertions pass: ``chebweights(n, kind=1)`` now returns Fejér's
first-rule weights (ported from ``@chebtech1/quadwts.m``) and
``_cheb1_barywts`` supplies the 1st-kind barycentric weights (ported from
``@chebtech1/barywts.m``, giving ``[-1/sqrt2, 1/sqrt2]`` for n=2).

Provenance
----------
MATLAB source : tests/chebtech1/test_chebpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.diffmat import _cheb1_barywts
from chebfunjax.utils.interpolation import cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts, chebweights

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS
KIND = 1


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _x(n):
    return np.asarray(chebpts(n, kind=KIND))


def _w(n):
    return np.asarray(chebweights(n, kind=KIND))


def _v(n):
    # 1st-kind barycentric weights (MATLAB @chebtech1/barywts.m).  For
    # n <= 1 the single node / empty set is degenerate; cheb_bary_weights
    # gives the standard v = [] (n=0) and v = [1] (n=1) shared by both kinds.
    if n <= 1:
        return np.asarray(cheb_bary_weights(n))
    return np.asarray(_cheb1_barywts(n))


class TestChebtech1Chebpts:
    def test_n0_empty(self):
        assert _x(0).size == 0 and _w(0).size == 0 and _v(0).size == 0

    def test_n1(self):
        assert _x(1)[0] == 0.0 and _w(1)[0] == 2.0 and _v(1)[0] == 1.0

    def test_n2_points(self):
        x = _x(2)
        assert x.shape == (2,)
        assert _ninf(x - np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])) < TOL

    def test_n2_weights(self):
        w = _w(2)
        assert w.shape == (2,)
        assert np.array_equal(w, np.array([1.0, 1.0]))

    def test_n2_baryweights(self):
        v = _v(2)
        assert v.shape == (2,)
        assert _ninf(v - np.array([-1 / np.sqrt(2), 1 / np.sqrt(2)])) < TOL

    def test_n3_points(self):
        x = _x(3)
        assert x.shape == (3,)
        assert _ninf(x - np.array([-np.sqrt(3) / 2, 0.0, np.sqrt(3) / 2])) < TOL

    def test_n3_weights(self):
        w = _w(3)
        assert w.shape == (3,)
        assert _ninf(w - np.array([4 / 9, 10 / 9, 4 / 9])) < TOL

    def test_n3_baryweights(self):
        v = _v(3)
        assert v.shape == (3,)
        assert _ninf(v - np.array([0.5, -1.0, 0.5])) < TOL

    def test_n129_sizes(self):
        n = 129
        assert _x(n).shape == (n,) and _w(n).shape == (n,) and _v(n).shape == (n,)

    def test_n129_symmetric_nodes(self):
        n = 129
        x = _x(n)
        left = x[: (n - 1) // 2]
        right = x[(n + 1) // 2 :][::-1]
        assert _ninf(left + right) < TOL

    def test_n129_center_node(self):
        n = 129
        assert _x(n)[(n - 1) // 2] < TOL
