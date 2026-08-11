"""Port of MATLAB Chebfun tests/linop/test_concatenation.m (Fable 5).

Uses the chebcolloc2 discretization, as the MATLAB test does.

Provenance
----------
MATLAB source : tests/linop/test_concatenation.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

import chebfunjax as cj
from chebfunjax.operators.blocks import D, I, mult
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)


class TestLinopConcatenation:
    def test_all_matlab_assertions(self):
        Id = I((0.0, 2.0))
        Dop = D((0.0, 1.0, 2.0))
        x = cj.chebfun(lambda t: t, domain=(0.0, 2.0))
        X = mult(x)

        A = Id + 2 * X
        B = Dop ** 2

        C = ChebMatrix([[A, A]])
        assert C.dense(5).shape == (5, 10)

        C = ChebMatrix([[A], [A]])
        assert C.dense(5).shape == (10, 5)

        C = ChebMatrix([[A, B]])
        assert C.dense([5, 5]).shape == (10, 20)

        C = ChebMatrix([[A], [B]])
        assert C.dense([5, 5]).shape == (20, 10)

        C = ChebMatrix([[A, B]])
        ChebMatrix([[A, B], [2 * A, 2 * B]])
        assert C.dense([5, 5]).shape == (10, 20)

        C = ChebMatrix([[A, x]])
        assert C.dense(5).shape == (5, 6)

        ChebMatrix([[A, x], [A, x]])
        assert C.dense(5).shape == (5, 6)

        x = cj.chebfun(lambda t: t, domain=(0.0, 0.5, 2.0))
        C = ChebMatrix([[x, Dop]])
        assert C.dense([5, 5, 5]).shape == (15, 16)
