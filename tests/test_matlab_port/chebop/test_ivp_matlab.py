"""Port of MATLAB Chebfun tests/chebop/test_ivp.m.

Restore the original Chebfun L2 norm assertion and literal tolerance 1e-10.
The prior port used a 40-point maximum error at 1e-8.

Provenance
----------
MATLAB source : tests/chebop/test_ivp.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

from math import exp

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-10


class TestChebopIvp:
    def test_linear_ivp_exact_solution(self):
        domain = (-1.0, 1.0)
        x = cj.chebfun(lambda t: t, domain=domain)
        problem = Chebop(lambda t, u: u.diff() - u, domain=domain)
        problem.lbc = exp(-1.0) - 1.0
        solution = problem.solve(1.0 - x)
        error = solution - (x.exp() + x)
        assert float(error.norm(2)) < TOL
