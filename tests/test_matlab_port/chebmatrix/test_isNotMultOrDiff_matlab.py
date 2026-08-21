"""Port of MATLAB Chebfun tests/chebmatrix/test_isNotMultOrDiff.m
(Fable 5).

Provenance
----------
MATLAB source : tests/chebmatrix/test_isNotMultOrDiff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.blocks import (
    primitive_functionals,
    primitive_operators,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

DOM = (0.0, 2.0)


class TestChebmatrixIsnotmultordiff:
    def test_all_matlab_assertions(self):
        z, e, s, dt = primitive_functionals(DOM)
        ZZ, I, DD, C, M = primitive_operators(DOM)
        sinf = cj.chebfun(lambda x: jnp.sin(x), domain=DOM)
        Mb = M(sinf)

        A = ChebMatrix([[ZZ, DD], [C, Mb]])
        assert A.is_not_diff_or_int == [[True, False],
                                        [False, True]]  # pass(1)
        B = ChebMatrix([[I], [s]])
        assert B.is_not_diff_or_int == [[True], [False]]  # pass(2)
        C2 = 2 * A
        assert C2.is_not_diff_or_int == [[True, False],
                                         [False, True]]  # pass(3)
        D2 = B + B
        assert D2.is_not_diff_or_int == [[True], [False]]  # pass(4)
        E2 = A * A
        assert E2.is_not_diff_or_int == [[False, False],
                                         [False, False]]  # pass(5)
