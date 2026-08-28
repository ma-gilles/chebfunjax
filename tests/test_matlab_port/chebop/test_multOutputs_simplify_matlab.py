"""Port of MATLAB Chebfun tests/chebop/test_multOutputs_simplify.m (Fable 5).

MATLAB's nargout-driven simplification maps to
``SystemSolution.deal()`` (per-component simplify) vs direct indexing
(components share the solve grid).

Provenance
----------
MATLAB source : tests/chebop/test_multOutputs_simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)


def _len(u):
    return sum(p.tech.coeffs.shape[0] for p in u.funs)


class TestChebopMultOutputsSimplify:
    def test_all_matlab_assertions(self):
        L = Chebop(lambda t, u, v: [u.diff(2) + u,
                                    v.diff(2) + 100.0 * v],
                   domain=(0.0, 10.0))
        L.lbc = lambda u, v: [u - 1.0, u.diff(), v - 1.0, v.diff()]
        uv = L.solve(0.0)
        u, v = uv.deal()
        assert _len(u) != _len(v)                 # pass(1)
        assert _len(uv[0]) == _len(uv[1])         # pass(2)
        u2, v2 = uv.deal()
        assert _len(u) == _len(u2) and _len(v) == _len(v2)  # pass(3)
