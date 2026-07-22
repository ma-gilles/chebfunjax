"""Port of MATLAB Chebfun tests/chebop/test_jump_scaled.m (Fable 5).

A scaled interior jump condition (Chebfun issue #1694): the ``.bc`` imposes
``u(0.1^-) - 4 u(0.1^+) = 2.2`` together with a continuous first derivative,
so the solution jumps by a scaled amount at the interior point 0.1.

Provenance
----------
MATLAB source : tests/chebop/test_jump_scaled.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from chebfunjax.chebfun1d.chebfun import jump  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402


class TestChebopJumpScaled:
    def test_scaled_jump(self):
        A = Chebop(lambda x, u: u.diff(2) - u + x, (-1.0, 1.0))
        A.lbc = 0.2
        A.rbc = 0.0
        A.bc = lambda x, u: [
            u(0.1, "left") - 4 * u(0.1, "right") - 2.2,   # scaled jump in sol
            jump(u.diff(), 0.1, 1),                        # continuous 1st deriv
        ]
        u = A.solve(0.0)
        scaled_jump = float(u(0.1, "left")) - 4 * float(u(0.1, "right")) - 2.2
        assert abs(scaled_jump) < 1e-10
