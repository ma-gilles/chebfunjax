"""Port of MATLAB Chebfun tests/chebop/test_linearizationDimensions.m (Fable 5).

Tests that linearizing a (possibly multi-variable) CHEBOP produces a block
operator whose blocks carry the correct *type*: an ``OperatorBlock`` for an
unknown that is differentiated/integrated somewhere in the system, and a
``Chebfun`` (its multiplicative coefficient) for a pure-parameter unknown that
never is.  Mirrors the ``isParam`` bookkeeping of MATLAB
``@chebop/linearize.m``::

    isParam = any(any(~isNotDiffOrInt)) & all(isNotDiffOrInt, 1);

``chebfunjax`` exposes this via :meth:`Chebop.linop`, returning a ``ChebMatrix``
whose ``blocks`` are typed exactly as MATLAB ``linop(N).blocks``.

Provenance
----------
MATLAB source : tests/chebop/test_linearizationDimensions.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.blocks import OperatorBlock
from chebfunjax.operators.chebop import Chebop


def _is_op(block) -> bool:
    return isinstance(block, OperatorBlock)


def _is_chebfun(block) -> bool:
    return isinstance(block, Chebfun)


class TestChebopLinearizationdimensions:
    def test_all_matlab_assertions(self):
        # Example 1: L = chebop(@(x,u,v) u + v);
        # No variable gets differentiated or integrated, so both blocks are
        # operatorBlocks.
        blocks = Chebop(lambda x, u, v: u + v).linop()
        assert _is_op(blocks[0]) and _is_op(blocks[1])

        # Example 2: L = chebop(@(x,u) u{1} + u{2});  (MATLAB cell form)
        # Two undifferentiated unknowns -> both operatorBlocks.  chebfunjax
        # writes a two-unknown system with explicit arguments.
        blocks = Chebop(lambda x, u, v: u + v).linop()
        assert _is_op(blocks[0]) and _is_op(blocks[1])

        # Example 3: L = chebop(@(x,u,v) diff(u) + v);
        # u is differentiated (operatorBlock); v is a parameter (chebfun).
        blocks = Chebop(lambda x, u, v: u.diff() + v).linop()
        assert _is_op(blocks[0]) and _is_chebfun(blocks[1])

        # Example 4: L = chebop(@(x,u,v) u + diff(v));
        # u is a parameter (chebfun); v is differentiated (operatorBlock).
        blocks = Chebop(lambda x, u, v: u + v.diff()).linop()
        assert _is_chebfun(blocks[0]) and _is_op(blocks[1])

        # Example 5: L = chebop(@(x,u,v) diff(u) + diff(v));
        # Both unknowns are differentiated -> both operatorBlocks.
        blocks = Chebop(lambda x, u, v: u.diff() + v.diff()).linop()
        assert _is_op(blocks[0]) and _is_op(blocks[1])

        # Example 6: L = chebop(@(x,u,v) [u + diff(v); diff(u) + v]);
        # u is differentiated in row 2, v in row 1: every unknown is
        # differentiated somewhere, so all four blocks are operatorBlocks.
        blocks = Chebop(lambda x, u, v: [u + v.diff(), u.diff() + v]).linop()
        assert blocks.size == (2, 2)
        assert (_is_op(blocks[(0, 0)]) and _is_op(blocks[(1, 0)])
                and _is_op(blocks[(0, 1)]) and _is_op(blocks[(1, 1)]))

        # Example 7: L = chebop(@(x,u,v,w) u + diff(v) + w);
        # Only v is differentiated; u and w are parameters (chebfuns).
        blocks = Chebop(lambda x, u, v, w: u + v.diff() + w).linop()
        assert (_is_chebfun(blocks[0]) and _is_op(blocks[1])
                and _is_chebfun(blocks[2]))

        # Example 8: L = chebop(@(x,u,v,w) [u + diff(v) + w; u + v + diff(w)]);
        # u is never differentiated (chebfun column); v and w are
        # (operatorBlock columns).
        blocks = Chebop(
            lambda x, u, v, w: [u + v.diff() + w, u + v + w.diff()]
        ).linop()
        assert blocks.size == (2, 3)
        assert (_is_chebfun(blocks[(0, 0)]) and _is_chebfun(blocks[(1, 0)])
                and _is_op(blocks[(0, 1)]) and _is_op(blocks[(1, 1)])
                and _is_op(blocks[(0, 2)]) and _is_op(blocks[(1, 2)]))
