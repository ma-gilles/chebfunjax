"""Core tests for fun-level empty representations (Fable 5).

MATLAB's ``bndfun()`` and ``deltafun()`` with no arguments give empty objects:
``isempty`` is True and arithmetic / restriction / calculus with an empty
operand propagates the empty.  A Deltafun is empty iff it has no deltas AND its
funPart is empty.
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun

DOM = Domain((-2.0, 7.0))
D1 = Domain((-1.0, 1.0))


class TestBndfunEmpty:
    def test_empty_is_empty(self):
        assert Bndfun.empty().isempty()

    def test_constructed_not_empty(self):
        assert not Bndfun.from_function(jnp.sin, DOM).isempty()

    def test_arithmetic_propagates(self):
        e = Bndfun.empty()
        g = Bndfun.from_function(jnp.sin, DOM)
        assert (e + g).isempty()
        assert (g + e).isempty()
        assert (e - g).isempty()
        assert (g * e).isempty()
        assert (2.0 * e).isempty()

    def test_restrict_propagates(self):
        assert Bndfun.empty().restrict(-1.0, 1.0).isempty()


class TestDeltafunEmpty:
    def test_marker_empty(self):
        assert Deltafun.empty().isempty()

    def test_empty_funpart_is_empty(self):
        assert Deltafun.from_fun(Bndfun.empty()).isempty()

    def test_smooth_deltafun_not_empty(self):
        d = Deltafun.from_fun(Bndfun.from_function(jnp.sin, D1))
        assert not d.isempty()

    def test_ops_propagate(self):
        e = Deltafun.empty()
        d = Deltafun(Bndfun.from_function(jnp.sin, D1),
                     jnp.array([0.0]), jnp.array([1.0]))
        assert (e + d).isempty()
        assert (d + e).isempty()
        assert (e * d).isempty()
        assert (d * e).isempty()
        assert e.diff().isempty()
        assert e.cumsum().isempty()
        assert e.real().isempty()
        assert e.imag().isempty()
        assert e.restrict([-0.5, 0.5]).isempty()

    def test_minandmax_empty(self):
        vals, pos = Deltafun.empty().minandmax()
        assert vals.shape[0] == 0 and pos.shape[0] == 0
