"""Port of MATLAB Chebfun tests/chebop/test_pantograph.m (Fable 5).

Pantograph-type (functional / delay) equations: the unknown is
evaluated at a chebfun argument inside the operator, ``u(x/2)``,
``u(u)``, ``u(.5*u')``, ``u(x*u^2)`` -- chebfun composition.

Provenance
----------
MATLAB source : tests/chebop/test_pantograph.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

jax.config.update("jax_enable_x64", True)

TOL = 1e-10
D = (0.0, 2.0)


def _n2(f):
    return float(f.norm(2))


class TestChebopPantograph:
    def test_all_matlab_assertions(self):
        x = cj.chebfun(lambda t: t, domain=D)

        N = Chebop(lambda x, u: u.diff(2) - u(x / 2.0), domain=D)
        N.lbc = 0.0
        N.rbc = 1.0
        u = N.solve(0.0)
        assert _n2(u.diff(2) - u(x / 2.0)) < TOL              # pass(1)

        N = Chebop(lambda x, u: u.diff(2) - u(u), domain=D)
        N.lbc = 0.0
        N.rbc = 1.0
        u = N.solve(0.0)
        assert _n2(u.diff(2) - u(u)) < TOL                    # pass(2)

        N = Chebop(lambda x, u: u.diff(2) - u(0.5 * u.diff()), domain=D)
        N.lbc = 0.0
        N.rbc = 1.0
        u = N.solve(0.0)
        assert _n2(u.diff(2) - u(0.5 * u.diff())) < TOL       # pass(3)

        N = Chebop(lambda x, u: u.diff(2) - u(x * u ** 2), domain=D)
        N.lbc = 0.1
        N.rbc = 1.0
        u = N.solve(0.0)
        assert _n2(u.diff(2) - u(x * u ** 2)) < 100 * TOL     # pass(4)
