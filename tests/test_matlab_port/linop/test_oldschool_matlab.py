"""Port of MATLAB Chebfun tests/linop/test_oldschool.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_oldschool.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    D,
    I,
    cumsum_op,
    diag,
    eval_at,
    sum_functional,
    zeros_op,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)


def _does_not_crash(fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


class TestLinopOldschool:
    def test_all_matlab_assertions(self):
        d = (-1.0, 1.0)

        # Just test whether these will execute.
        assert _does_not_crash(lambda: D(d))
        assert _does_not_crash(lambda: I(d))
        assert _does_not_crash(
            lambda: diag(cj.chebfun(lambda t: t, domain=d), d))
        assert _does_not_crash(lambda: cumsum_op(d))
        assert _does_not_crash(lambda: sum_functional(d))
        assert _does_not_crash(lambda: eval_at(0.0, d, "left"))
        assert _does_not_crash(lambda: zeros_op(d))

        # The feval-style instantiation shorthand.
        Dop = D(d)
        assert float(jnp.linalg.norm(
            Dop.matrix(8) - ChebMatrix([[Dop]]).dense(8))) < 2e-14

        # Boundary condition syntax: classic row replacement.
        A = Dop ** 2
        A0 = ChebMatrix([[A]]).dense(12)         # version with no BCs
        L = linop(A)
        L = L.add_constraint(eval_at(d[0], d), 0.0)
        L = L.add_constraint(eval_at(d[-1], d) * Dop, 0.0)
        A1 = L.matrix(10)                        # first two rows hold BCs
        correct = A0.at[0, :].set(A1[0, :]).at[-1, :].set(A1[1, :])

        a_old = L.matrix_oldschool(12)
        assert float(jnp.linalg.norm(correct - a_old)) < 2e-14
