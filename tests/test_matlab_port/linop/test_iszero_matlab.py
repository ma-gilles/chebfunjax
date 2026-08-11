"""Port of MATLAB Chebfun tests/linop/test_iszero.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_iszero.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.operators.blocks import D, I, zeros_op
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)


class TestLinopIszero:
    def test_all_matlab_assertions(self):
        d = (0.0, 1.0)
        Id = I(d)
        Dop = D(d)
        Z = zeros_op(d)

        # Concatenations.
        A = ChebMatrix([[Id, Dop], [Id, Z]])
        assert A.iszero() == [[0, 0], [0, 1]]

        # Plus.
        B1 = Id + Z
        assert not B1.iszero

        B2 = ChebMatrix([[Id, Z]]) + ChebMatrix([[Id, Dop]])
        assert not any(B2.iszero()[0])

        B3 = ChebMatrix([[Z, Z]]) + ChebMatrix([[Z, Dop]])
        assert B3.iszero() == [[1, 0]]

        # Times.
        C1 = ChebMatrix([[Id]]) * ChebMatrix([[Dop, Z]])
        assert C1.iszero() == [[0, 1]]

        C2 = ChebMatrix([[Id], [Id]]) * ChebMatrix([[Z]])
        assert C2.iszero() == [[1], [1]]

        # Scalars.
        E1 = Dop * 0
        assert E1.iszero

        E2 = Z + 2
        assert not E2.iszero
