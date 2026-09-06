"""Port of MATLAB Chebfun tests/functionalBlock/test_isNotMultOrDiff.m
(Fable 5).  MATLAB's ``isNotDiffOrInt`` flag is ``isnotdiffint``.

Provenance
----------
MATLAB source : tests/functionalBlock/test_isNotMultOrDiff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.blocks import primitive_functionals, primitive_operators

jax.config.update("jax_enable_x64", True)


class TestFunctionalBlockIsNotMultOrDiff:
    def test_all_matlab_assertions(self):
        dom = (0.0, 2.0)
        Z, E, S, D = primitive_functionals(dom)
        E = E(1.0)
        D = D(chebfun(jnp.sin, domain=dom))
        ZZ, Id, DD, C, M = primitive_operators(dom)
        M = M(chebfun(jnp.sin, domain=dom))
        assert Z.isnotdiffint is True                                        # pass(1)
        assert E.isnotdiffint is True                                        # pass(2)
        assert S.isnotdiffint is False                                       # pass(3)
        assert D.isnotdiffint is True                                        # pass(4)
        assert (Z + Id).isnotdiffint is True                                 # pass(5)
        assert (E + S).isnotdiffint is False                                 # pass(6)
        assert (2 * Z).isnotdiffint is True                                  # pass(7)
        assert (2 * S).isnotdiffint is False                                 # pass(8)
        assert (S * ZZ).isnotdiffint is True                                 # pass(9)
        assert (D * DD).isnotdiffint is False                                # pass(10)
        assert (E * C).isnotdiffint is False                                 # pass(11)
