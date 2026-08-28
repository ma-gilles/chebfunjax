"""Port of MATLAB Chebfun tests/chebop/test_multipleOutputs.m (Fable 5).

MATLAB's ``[u, info] = N\\f`` maps to ``N.solve_with_info(f)``; system
solutions unpack as Python sequences.

Provenance
----------
MATLAB source : tests/chebop/test_multipleOutputs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.chebfun1d.chebfun import Chebfun
from chebfunjax.operators.chebop import Chebop, SystemSolution

jax.config.update("jax_enable_x64", True)

D = (0.0, 2.0)


class TestChebopMultipleOutputs:
    def test_scalar_outputs(self):
        N = Chebop(lambda x, u: u.diff(2) + 0.01 * u.sin(), domain=D)
        N.lbc = 0.0
        N.rbc = 2.0
        u = N.solve(0.0)
        assert isinstance(u, Chebfun)                     # pass(1)
        u, info = N.solve_with_info(0.0)
        assert isinstance(u, Chebfun) and isinstance(info, dict)  # 2

        N = Chebop(lambda x, u: u.diff(2) + u.sin(), domain=D)
        N.lbc = lambda u: [u, u.diff() - 1.0]
        u = N.solve(0.0)
        assert isinstance(u, Chebfun)                     # pass(3)
        N = Chebop(lambda x, u: u.diff(2) + u.sin(), domain=D)
        N.lbc = lambda u: [u, u.diff() - 1.0]
        u, info = N.solve_with_info(0.0)
        assert isinstance(u, Chebfun) and isinstance(info, dict)  # 4

    def test_system_outputs(self):
        def mk():
            N = Chebop(lambda x, u, v: [u.diff(2) + 0.1 * u.sin() + v,
                                        v.diff(2) + 0.1 * v.cos() + u],
                       domain=D)
            N.lbc = lambda u, v: [u - 1.0, v.diff()]
            N.rbc = lambda u, v: [v.diff(), v + 1.0]
            return N

        uv = mk().solve(0.0)
        assert isinstance(uv, SystemSolution)             # pass(5)
        u, v = mk().solve(0.0)
        assert isinstance(u, Chebfun) and isinstance(v, Chebfun)  # 6
        sol, info = mk().solve_with_info(0.0)
        u, v = sol
        assert (isinstance(u, Chebfun) and isinstance(v, Chebfun)
                and isinstance(info, dict))               # pass(7)
