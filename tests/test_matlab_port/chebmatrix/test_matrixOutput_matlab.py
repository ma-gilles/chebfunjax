"""Port of MATLAB Chebfun tests/chebmatrix/test_matrixOutput.m
(Fable 5).

Chebmatrix operations resulting only in doubles return a plain
matrix; operator applications return chebfun blocks.

Provenance
----------
MATLAB source : tests/chebmatrix/test_matrixOutput.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.blocks import (
    primitive_functionals,
    primitive_operators,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

DOM = (-3.0, 1.0)


def _chebpoly(k):
    """The degree-k Chebyshev polynomial on DOM (MATLAB chebpoly)."""
    return cj.chebfun(
        lambda x: np.cos(k * np.arccos(np.clip(
            (2 * x - (DOM[0] + DOM[1])) / (DOM[1] - DOM[0]), -1, 1))),
        domain=DOM)


class TestChebmatrixMatrixoutput:
    def test_all_matlab_assertions(self):
        T = [_chebpoly(k) for k in range(1, 6)]
        # T'*T: 5x5 numeric Gram matrix.  pass(1)
        Gm = ChebMatrix([[t.transpose()] for t in T]) * \
            ChebMatrix([T])
        assert hasattr(Gm, "shape") and Gm.shape == (5, 5)

        # Operators applied to chebfuns give chebfun blocks.  pass(3)
        Z, I, D, Cop, M = primitive_operators(DOM)
        A = ChebMatrix([[I, D], [Z, Cop]])
        f = ChebMatrix([[_chebpoly(1)], [_chebpoly(5)]])
        Af = A * f
        assert isinstance(Af, ChebMatrix) and Af.size == (2, 1)
        assert isinstance(Af[0, 0], Chebfun)
        assert isinstance(Af[1, 0], Chebfun)

        # Functionals give numbers.  pass(4)/(5)
        z, e, s, dt = primitive_functionals(DOM)
        J = ChebMatrix([[z - e(1.0), s]])
        Jf = J * f
        assert hasattr(Jf, "shape") and Jf.shape == (1, 1)
        K = ChebMatrix([[z - e(1.0), s], [z - e(1.0), s]])
        Kf = K * f
        assert hasattr(Kf, "shape") and Kf.shape == (2, 1)
