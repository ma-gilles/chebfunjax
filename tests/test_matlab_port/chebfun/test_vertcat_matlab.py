"""Port of MATLAB Chebfun tests/chebfun/test_vertcat.m (Fable 5).

FIXED (Fable 5): [f; g] block concatenation via ChebMatrix.vertcat /
horzcat (a 2-D block container with a ``.blocks`` cell array).

All seven MATLAB assertions port: pass(5,6) use the row/column
orientation flag (is_transposed) — [x'; x'] builds an array-valued row
chebfun and mixing orientations raises.

Provenance
----------
MATLAB source : tests/chebfun/test_vertcat.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain
from chebfunjax.operators.chebmatrix import ChebMatrix


class TestChebfunVertcat:
    def test_all_matlab_assertions(self):
        x = Chebfun.from_function(lambda t: t, Domain((-1.0, 1.0)))

        # pass(1): [x; 1] -> 2x1 chebmatrix, blocks{2,1} numeric.
        f = ChebMatrix.vertcat(x, 1)
        assert f.size == (2, 1)
        assert isinstance(f.blocks[1][0], (int, float))

        # pass(2): [x; x] -> 2x1 chebmatrix, blocks{2,1} a chebfun.
        f = ChebMatrix.vertcat(x, x)
        assert f.size == (2, 1)
        assert isinstance(f.blocks[1][0], Chebfun)

        # pass(3): [x 1; x 1] -> 2x2, mixed block types.
        f = ChebMatrix.vertcat(ChebMatrix.horzcat(x, 1),
                               ChebMatrix.horzcat(x, 1))
        assert f.size == (2, 2)
        assert isinstance(f.blocks[1][0], Chebfun)
        assert isinstance(f.blocks[0][1], (int, float))

        # pass(4): [x abs(x); x x] -> 2x2 of chebfuns.
        ax = Chebfun.from_function(lambda t: abs(t), Domain((-1.0, 1.0)))
        f = ChebMatrix.vertcat(ChebMatrix.horzcat(x, ax),
                               ChebMatrix.horzcat(x, x))
        assert f.size == (2, 2)
        assert isinstance(f.blocks[1][0], Chebfun)
        assert isinstance(f.blocks[0][1], Chebfun)

        # pass(5): row chebfuns [x'; x'] -> array-valued chebfun with
        # 2 columns.
        xt = x.transpose()
        f = Chebfun.vertcat([xt, xt])
        assert isinstance(f, Chebfun)
        assert f.n_columns == 2

        # pass(6): mixing a column with a row is an error.
        import pytest
        with pytest.raises(ValueError):
            Chebfun.vertcat([x, xt])

        # pass(7): [x; [0; 0]] -> 3x1 chebmatrix.
        f = ChebMatrix.vertcat(x, np.array([0.0, 0.0]))
        assert f.size == (3, 1)
