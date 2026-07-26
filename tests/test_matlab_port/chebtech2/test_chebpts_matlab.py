"""Port of MATLAB Chebfun tests/chebtech2/test_chebpts.m (Opus 4.8).

Self-validating: 2nd-kind Chebyshev points, Clenshaw-Curtis quadrature
weights and barycentric weights are checked against closed-form exacts at
the SAME tolerance MATLAB uses (10*eps for the tolerance checks; exact
equality where MATLAB uses ``==``).

MATLAB's ``chebtech2.chebpts(n)`` returns ``[x, w, v]``.  In chebfunjax:
    x = chebpts(n, kind=2)
    w = chebweights(n, kind=2)          (Clenshaw-Curtis, matches MATLAB)
    v = cheb_bary_weights(n)            (2nd-kind barycentric, matches MATLAB)
chebfunjax returns 1-D arrays of shape ``(n,)`` throughout, so size checks
are adapted to ``.shape == (n,)``.

All assertions pass, including the three exact-``==`` centre/antisymmetry
checks: ``chebpts(n, kind=2)`` now uses MATLAB's sine construction
``x = sin(pi*(-m:2:m)/(2m))`` (m = n-1), which is bit-exactly antisymmetric
(centre node and antisymmetry residual are exactly ``0``).

Provenance
----------
MATLAB source : tests/chebtech2/test_chebpts.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils.interpolation import cheb_bary_weights
from chebfunjax.utils.quadrature import chebpts, chebweights

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS
KIND = 2


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _x(n):
    return np.asarray(chebpts(n, kind=KIND))


def _w(n):
    return np.asarray(chebweights(n, kind=KIND))


def _v(n):
    # 2nd-kind barycentric weights (MATLAB chebtech2 barywts).
    return np.asarray(cheb_bary_weights(n))


class TestChebtech2Chebpts:
    def test_n0_empty(self):
        assert _x(0).size == 0 and _w(0).size == 0 and _v(0).size == 0

    def test_n1(self):
        assert _x(1)[0] == 0.0 and _w(1)[0] == 2.0 and _v(1)[0] == 1.0

    def test_n2_points(self):
        x = _x(2)
        assert x.shape == (2,)
        assert np.array_equal(x, np.array([-1.0, 1.0]))

    def test_n2_weights(self):
        w = _w(2)
        assert w.shape == (2,)
        assert np.array_equal(w, np.array([1.0, 1.0]))

    def test_n2_baryweights(self):
        v = _v(2)
        assert v.shape == (2,)
        assert np.array_equal(v, 0.5 * np.array([-1.0, 1.0]))

    def test_n3_points(self):
        x = _x(3)
        assert x.shape == (3,)
        assert np.array_equal(x, np.array([-1.0, 0.0, 1.0]))

    def test_n3_weights(self):
        w = _w(3)
        assert w.shape == (3,)
        assert _ninf(w - (np.array([0.0, 1.0, 0.0]) + 1 / 3)) < TOL

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
        assert _ninf(left + right) == 0.0

    def test_n129_center_node(self):
        n = 129
        assert _x(n)[(n - 1) // 2] == 0.0
