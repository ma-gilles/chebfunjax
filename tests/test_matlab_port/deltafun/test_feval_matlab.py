"""Port of MATLAB Chebfun tests/deltafun/test_feval.m (Opus 4.8).

In MATLAB, evaluating a Deltafun *at* a delta location returns +/-Inf (with the
sign of the delta magnitude).  chebfunjax's ``Deltafun.__call__`` deliberately
evaluates only the smooth ``funPart`` (deltas are distributional and have no
pointwise value), so it returns a finite number at delta locations.  Those
assertions are captured as xfails.  The later assertions exercise chebfun-level
``dirac``/``heaviside`` and directional (``'left'``/``'right'``) evaluation,
which live above the fun/deltafun layer and are skipped.

Provenance
----------
MATLAB source : tests/deltafun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DOM = Domain((-1.0, 1.0))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestDeltafunFeval:
    def test_positive_delta_is_plus_inf(self):
        # pass(1): isinf(feval(d,0)) && feval(d,0) > 0
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        val = float(d(jnp.float64(0.0)))
        assert np.isinf(val) and val > 0

    def test_negated_delta_is_minus_inf(self):
        # pass(2, first): isinf(feval(-d,0)) && feval(-d,0) < 0
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun(f, jnp.array([0.0]), jnp.array([1.0]))
        val = float((-d)(jnp.float64(0.0)))
        assert np.isinf(val) and val < 0

    def test_empty_delta_matches_funpart(self):
        # pass(2, second): norm(feval(f,x) - feval(d,x), inf) == 0 (no deltas)
        f = Bndfun.from_function(jnp.sin, DOM)
        d = Deltafun.from_fun(f)
        x = jnp.asarray(np.linspace(-0.9, 0.9, 4))
        assert _ninf(f(x) - d(x)) == 0.0

    def test_eval_at_delta_locations_is_inf(self):
        # pass(3): all(isinf(feval(d,x))) where deltas sit at x
        f = Bndfun.from_function(jnp.sin, DOM)
        x = np.linspace(-0.8, 0.8, 4)
        d = Deltafun(f, jnp.asarray(x), jnp.asarray(np.random.rand(4)))
        vals = np.array(d(jnp.asarray(x)))
        assert np.all(np.isinf(vals))

    def test_chebfun_level_dirac_heaviside_directional(self):
        # pass(4)-(15): dirac(x-1), diff(heaviside(x)), 'left'/'right' evaluation
        from chebfunjax.chebfun1d.chebfun import chebfun
        x = chebfun(lambda t: t)
        d = (x - 1).dirac()
        assert np.isinf(float(d(1.0)))                                       # pass(4)
        val = float((-d)(1.0))
        assert np.isinf(val) and val < 0                                     # pass(5)
        assert np.isinf(float(d("right")))                                   # pass(6)
        assert float(d("left")) == 0                                         # pass(7)
        assert float(d(1.0, "left")) == 0                                    # pass(8)
        assert float(d(0.0, "left")) == 0                                    # pass(9)
        assert float(d(0.0, "right")) == 0                                   # pass(10)
        d = x.heaviside().diff()
        assert np.isinf(float(d(0.0)))                                       # pass(11)
        assert float(d("right")) == 0                                        # pass(12)
        assert float(d("left")) == 0                                         # pass(13)
        assert float(d(0.0, "left")) == 0                                    # pass(14)
        assert float(d(0.0, "right")) == 0                                   # pass(15)
