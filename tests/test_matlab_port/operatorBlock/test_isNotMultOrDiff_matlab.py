"""Port of MATLAB Chebfun tests/operatorBlock/test_isNotMultOrDiff.m
(Fable 5).  MATLAB's ``isNotDiffOrInt`` flag is ``isnotdiffint``.

Provenance
----------
MATLAB source : tests/operatorBlock/test_isNotMultOrDiff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import primitive_operators

jax.config.update("jax_enable_x64", True)


class TestOperatorBlockIsNotMultOrDiff:
    def test_all_matlab_assertions(self):
        dom = (0.0, 2.0)
        Z, I, D, C, M = primitive_operators(dom)
        M = M(chebfun(jnp.sin, domain=dom))
        assert Z.isnotdiffint is True                                        # pass(1)
        assert I.isnotdiffint is True                                        # pass(2)
        assert D.isnotdiffint is False                                       # pass(3)
        assert C.isnotdiffint is False                                       # pass(4)
        assert M.isnotdiffint is True                                        # pass(5)
        assert (Z + I).isnotdiffint is True                                  # pass(6)
        assert (Z + D).isnotdiffint is False                                 # pass(7)
        assert (M * Z).isnotdiffint is True                                  # pass(8)
        assert (C * Z).isnotdiffint is True                                  # pass(9)
        assert (C + M).isnotdiffint is False                                 # pass(10)
        assert (2 * M).isnotdiffint is True                                  # pass(11)
        assert (2 * D).isnotdiffint is False                                 # pass(12)
