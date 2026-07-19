"""Port of MATLAB Chebfun tests/deltafun/test_plus.m (Opus 4.8).

Adding two Deltafuns adds their funParts and merges their delta data, summing
magnitudes at coincident locations and sorting the result by location.  MATLAB
uses random delta locations; deterministic distinct locations are used here so
the exact-equality assertions hold.

Provenance
----------
MATLAB source : tests/deltafun/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

A, B = -4.0, 4.0
DAB = Domain((A, B))
X = jnp.asarray(np.linspace(A, B, 60))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestDeltafunPlus:
    def test_empty_plus_empty(self):
        # pass(1): isempty(deltafun() + deltafun())
        assert (Deltafun.empty() + Deltafun.empty()).isempty()

    def test_empty_plus_nonempty(self):
        # pass(2): isempty(deltafun()+df1) && isempty(df1+deltafun())
        df1 = Deltafun(Bndfun.from_function(jnp.sin, DAB),
                       jnp.array([0.5]), jnp.array([1.0]))
        assert (Deltafun.empty() + df1).isempty()
        assert (df1 + Deltafun.empty()).isempty()

    def test_same_locations_sum_and_sort(self):
        # pass(3): df1 + df2 with shared locations -> mags summed, sorted
        f1 = Bndfun.from_function(jnp.sin, DAB)
        f2 = Bndfun.from_function(jnp.cos, DAB)
        d1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        d2 = np.array([[-1.0, 0.5, 2.0], [3.0, -4.0, 1.0], [0.0, 6.0, -2.0]])
        l1 = np.array([1.5, -2.0, 3.2])  # distinct, unsorted
        df1 = Deltafun(f1, jnp.asarray(l1), jnp.asarray(d1))
        df2 = Deltafun(f2, jnp.asarray(l1), jnp.asarray(d2))
        s = df1 + df2
        idx = np.argsort(l1)
        A_exp = (d1 + d2)[:, idx]
        assert _ninf(s.delta_mags - A_exp) == 0.0
        assert _ninf(s.delta_locs - np.sort(l1)) == 0.0
        assert _ninf((f1 + f2)(X) - s.funPart(X)) == 0.0

    def test_doubling_identical_deltas(self):
        # pass(4): (f1, d @ l) + (f1, sort(d) @ sort(l)) -> 2*A, sorted, 2*f1
        f1 = Bndfun.from_function(jnp.sin, DAB)
        l = np.array([0.7, -0.3, 0.1, -0.9, 0.4])
        d = np.random.rand(3, 5)
        idx = np.argsort(l)
        sl = l[idx]
        A_exp = d[:, idx]
        s = (Deltafun(f1, jnp.asarray(l), jnp.asarray(d))
             + Deltafun(f1, jnp.asarray(sl), jnp.asarray(A_exp)))
        assert _ninf(s.delta_mags - 2 * A_exp) == 0.0
        assert _ninf(s.delta_locs - sl) == 0.0
        assert _ninf((2.0 * f1)(X) - s.funPart(X)) == 0.0

    def test_partial_overlap_merge(self):
        # pass(5): overlapping/disjoint locations merge, sort, sum magnitudes
        f1 = Bndfun.from_function(jnp.sin, DAB)
        f2 = Bndfun.from_function(jnp.cos, DAB)
        df1 = Deltafun(f1, jnp.array([-0.25, 0.5, -0.5, -0.8]),
                       jnp.array([1.0, 2.0, 3.0, 4.0]))
        df2 = Deltafun(f2, jnp.array([-0.25, 0.6, -0.5, -0.7]),
                       jnp.array([1.0, 2.0, 3.0, 4.0]))
        s = df1 + df2
        assert _ninf(s.delta_locs - np.array([-0.8, -0.7, -0.5, -0.25, 0.5, 0.6])) == 0.0
        assert _ninf(s.delta_mags - np.array([4.0, 4.0, 6.0, 2.0, 2.0, 2.0])) == 0.0
