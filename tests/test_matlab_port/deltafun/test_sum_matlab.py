"""Port of MATLAB Chebfun tests/deltafun/test_sum.m (Opus 4.8).

The definite integral of a Deltafun is the integral of its funPart plus the sum
of the order-0 delta magnitudes.  chebfunjax has no empty Deltafun, so the empty
case (sum == 0) is exercised via the zero distribution (zero funPart, no deltas).

Provenance
----------
MATLAB source : tests/deltafun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DELTA_TOL = 1e-9  # pref.deltaPrefs.deltaTol
DOM = Domain((-1.0, 1.0))


class TestDeltafunSum:
    def test_zero_distribution_sum_is_zero(self):
        # pass(1): sum(deltafun()) == 0  (zero distribution analog)
        d = Deltafun.from_fun(Bndfun.from_function(lambda x: jnp.zeros_like(x), DOM))
        assert float(d.sum()) == 0.0

    def test_sum_funpart_plus_row0_magnitudes(self):
        # pass(2): sum(d) == exp(1) - exp(-1) + sum(mag(1,:))
        f = Bndfun.from_function(jnp.exp, DOM)
        mag = np.random.rand(5, 5)
        loc = np.random.rand(5)
        d = Deltafun(f, jnp.asarray(loc), jnp.asarray(mag))
        expected = (np.exp(1.0) - np.exp(-1.0)) + mag[0].sum()
        assert abs(float(d.sum()) - expected) < DELTA_TOL

    def test_sum_cancelling_deltas(self):
        # pass(3): sin(pi*x) with deltas [-1, 1] at [-1, 1] -> sum == 0
        f = Bndfun.from_function(lambda x: jnp.sin(jnp.pi * x), DOM)
        d = Deltafun(f, jnp.array([-1.0, 1.0]), jnp.array([-1.0, 1.0]))
        assert abs(float(d.sum()) - 0.0) < DELTA_TOL
