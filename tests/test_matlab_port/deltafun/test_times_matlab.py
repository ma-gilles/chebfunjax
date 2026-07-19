"""Port of MATLAB Chebfun tests/deltafun/test_times.m (Fable 5).

A smooth function times a (possibly high-order) delta expands by the Leibniz
rule f(x)*delta^(m)(x - x0) into a weighted sum of lower-order deltas whose
magnitudes are signed derivatives of f at x0.  The two empty-Deltafun cases
(pass 1-2) are skipped: chebfunjax has no empty Deltafun.  Deterministic,
disjoint delta locations replace MATLAB's random ones.

Provenance
----------
MATLAB source : tests/deltafun/test_times.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DTOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))
A, B = -4.0, 4.0
DAB = Domain((A, B))
X = jnp.asarray(np.linspace(A + 0.05, B - 0.05, 50))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _dvals(fun, loc, k):
    """Evaluate the k-th derivative of a Bndfun at a scalar location."""
    g = fun
    for _ in range(k):
        g = g.diff()
    return float(g(jnp.float64(loc)))


class TestDeltafunTimes:
    def test_empty_times_empty(self):
        pytest.skip("chebfunjax has no empty Deltafun representation")

    def test_empty_times_delta(self):
        pytest.skip("chebfunjax has no empty Deltafun representation")

    def test_expneg_times_delta4(self):
        # pass(3): exp(-x) .* delta^(4) -> [1, 4, 6, 4, 1]'
        f = Bndfun.from_function(lambda x: jnp.exp(-x), DOM)
        g = Bndfun.from_function(jnp.sin, DOM)
        df1 = Deltafun.from_fun(f)
        df2 = Deltafun(g, jnp.array([0.0]),
                       jnp.array([[0.0], [0.0], [0.0], [0.0], [1.0]]))
        s = df1 * df2
        assert _ninf(s.delta_mags - np.array([[1.0], [4.0], [6.0], [4.0], [1.0]])) < DTOL

    def test_exp_times_delta3(self):
        # pass(4): exp(x) .* delta^(3) -> [-1, 3, -3, 1]'
        f = Bndfun.from_function(jnp.exp, DOM)
        g = Bndfun.from_function(jnp.sin, DOM)
        df1 = Deltafun.from_fun(f)
        df2 = Deltafun(g, jnp.array([0.0]),
                       jnp.array([[0.0], [0.0], [0.0], [1.0]]))
        s = df1 * df2
        assert _ninf(s.delta_mags - np.array([[-1.0], [3.0], [-3.0], [1.0]])) < DTOL

    # --- one delta block (single-block Leibniz) ---
    def _setup_single(self):
        f1 = Bndfun.from_function(lambda x: jnp.exp(jnp.sin(x)), DAB)
        f2 = Bndfun.from_function(lambda x: jnp.exp(jnp.cos(x)), DAB)
        d1 = 0.5 * np.array([[0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [1.0, 0.0, 1.0]])
        l1 = np.array([-3.0, -1.5, -0.5])
        df1 = Deltafun(f1, jnp.asarray(l1), jnp.asarray(d1))
        df2 = Deltafun.from_fun(f2)
        return f1, f2, l1, df1, df2

    def _deltas1(self, f2, l1):
        c1 = np.array([_dvals(f2, l1[0], 2), -2 * _dvals(f2, l1[0], 1),
                       _dvals(f2, l1[0], 0)])
        c2 = np.array([_dvals(f2, l1[1], 0) - _dvals(f2, l1[1], 1),
                       _dvals(f2, l1[1], 0), 0.0])
        c3 = np.array([_dvals(f2, l1[2], 2), -2 * _dvals(f2, l1[2], 1),
                       _dvals(f2, l1[2], 0)])
        return 0.5 * np.column_stack([c1, c2, c3])

    def test_single_funpart(self):
        # pass(5): iszero(s.funPart - f1.*f2)
        f1, f2, l1, df1, df2 = self._setup_single()
        s = df1 * df2
        assert _ninf(s.funPart(X) - (f1 * f2)(X)) < 1e3 * DTOL

    def test_single_locations(self):
        # pass(6): s.deltaLoc == sort(l1)
        _, _, l1, df1, df2 = self._setup_single()
        s = df1 * df2
        assert _ninf(s.delta_locs - np.sort(l1)) == 0.0

    def test_single_magnitudes(self):
        # pass(7): Leibniz magnitudes for f2 derivatives at l1
        _, f2, l1, df1, df2 = self._setup_single()
        s = df1 * df2
        assert _ninf(np.asarray(s.delta_mags) - self._deltas1(f2, l1)) < DTOL

    # --- two delta blocks ---
    def _setup_two(self):
        f1, f2, l1, df1, _ = self._setup_single()
        d2 = 0.8 * np.array([[1.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0]])
        l2 = np.array([0.5, 1.5, 3.0])
        df2 = Deltafun(f2, jnp.asarray(l2), jnp.asarray(d2))
        return f1, f2, l1, l2, df1, df2

    def _deltas2(self, f1, l2):
        c1 = np.array([_dvals(f1, l2[0], 0), 0.0, 0.0])
        c2 = np.array([_dvals(f1, l2[1], 2) - _dvals(f1, l2[1], 1),
                       _dvals(f1, l2[1], 0) - 2 * _dvals(f1, l2[1], 1),
                       _dvals(f1, l2[1], 0)])
        c3 = np.array([_dvals(f1, l2[2], 0), 0.0, 0.0])
        return 0.8 * np.column_stack([c1, c2, c3])

    def test_two_funpart(self):
        # pass(8): iszero(s.funPart - f1.*f2)
        f1, f2, l1, l2, df1, df2 = self._setup_two()
        s = df1 * df2
        assert _ninf(s.funPart(X) - (f1 * f2)(X)) < 1e3 * DTOL

    def test_two_locations(self):
        # pass(9): s.deltaLoc == sort(union(l1, l2))
        _, _, l1, l2, df1, df2 = self._setup_two()
        s = df1 * df2
        assert _ninf(s.delta_locs - np.sort(np.union1d(l1, l2))) == 0.0

    def test_two_magnitudes(self):
        # pass(10): [deltas1, deltas2] == s.deltaMag (l1 < l2 so order matches)
        f1, f2, l1, l2, df1, df2 = self._setup_two()
        s = df1 * df2
        expected = np.hstack([self._deltas1(f2, l1), self._deltas2(f1, l2)])
        assert _ninf(np.asarray(s.delta_mags) - expected) < DTOL
