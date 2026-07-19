"""Port of MATLAB Chebfun tests/deltafun/test_isempty.m (Fable 5).

A Deltafun is empty iff it has no deltas AND its funPart is empty
(``isempty(deltaLoc) && isempty(funPart)``).  The marker-empty ``Deltafun``
and a Deltafun wrapping an empty Bndfun are both empty.

Provenance
----------
MATLAB source : tests/deltafun/test_isempty.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun


class TestDeltafunIsempty:
    def test_empty_constructor(self):
        # pass(1): isempty(deltafun())
        assert Deltafun.empty().isempty()

    def test_empty_bndfun_funpart(self):
        # pass(2): isempty(deltafun(bndfun([])))
        assert Deltafun.from_fun(Bndfun.empty()).isempty()

    def test_empty_delta_arg(self):
        # pass(3): isempty(deltafun(f, [])) with empty funPart f
        d = Deltafun(Bndfun.empty(), [], [])
        assert d.isempty()

    def test_empty_delta_struct(self):
        # pass(4): isempty(deltafun(f, struct('deltaMag', [], 'deltaLoc', [])))
        import jax.numpy as jnp
        d = Deltafun(Bndfun.empty(), jnp.zeros(0), jnp.zeros((1, 0)))
        assert d.isempty()
