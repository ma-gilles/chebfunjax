"""Port of MATLAB Chebfun tests/chebop/test_periodic_nonlin.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebop/test_periodic_nonlin.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from chebfunjax.chebfun1d.chebfun import chebfun  # noqa: E402
from chebfunjax.operators.chebop import Chebop  # noqa: E402
from chebfunjax.tech.trigtech import Trigtech  # noqa: E402

TOL = 1e-8
DOM = (-math.pi, math.pi)


class TestChebopPeriodicNonlin:
    def test_second_order_vs_explicit_bc(self):
        # u'' - sin(u) = cos(2x), periodic tag vs explicit periodic bc
        f = chebfun(lambda x: jnp.cos(2.0 * x), domain=DOM)
        N = Chebop(lambda u: u.diff(2) - u.sin(), domain=DOM)
        N.bc = "periodic"
        N.init = f
        u = N.solvebvp(f)[0]

        N2 = Chebop(lambda u: u.diff(2) - u.sin(), domain=DOM)
        N2.bc = lambda x, u: [u(DOM[1]) - u(DOM[0]),
                              u.diff()(DOM[1]) - u.diff()(DOM[0])]
        N2.init = f
        v = N2.solve(f)
        assert float((u - v).norm(jnp.inf)) < TOL       # pass(1)

    def test_trigcolloc_first_order(self):
        # u' - u*cos(u) = cos(x)
        f = chebfun(lambda x: jnp.cos(x), domain=DOM)
        N = Chebop(lambda u: u.diff() - u * u.cos(), domain=DOM)
        N.bc = "periodic"
        N.init = f
        u = N.solve(f)
        assert float((N(u) - f).norm(2)) < TOL          # pass(2)
        assert abs(float(u(DOM[0])) - float(u(DOM[1]))) < TOL  # pass(3)
        assert isinstance(u.funs[0].tech, Trigtech)     # pass(4)

    def test_trigcolloc_second_order(self):
        # u'' - u^2*cos(u) = cos(x)
        f = chebfun(lambda x: jnp.cos(x), domain=DOM)
        N = Chebop(lambda u: u.diff(2) - u ** 2 * u.cos(), domain=DOM)
        N.bc = "periodic"
        N.init = f
        u = N.solve(f)
        assert float((N(u) - f).norm(2)) < TOL          # pass(5)
        assert abs(float(u(DOM[0])) - float(u(DOM[1]))) < TOL  # pass(6)
        du = u.diff()
        assert abs(float(du(DOM[0])) - float(du(DOM[1]))) < TOL  # pass(7)
        assert isinstance(u.funs[0].tech, Trigtech)     # pass(8)

    def test_trigspec_second_order(self):
        # Same problem under the coefficient discretization preference.
        f = chebfun(lambda x: jnp.cos(x), domain=DOM)
        N = Chebop(lambda u: u.diff(2) - u ** 2 * u.cos(), domain=DOM)
        N.bc = "periodic"
        N.init = f
        u = N.solvebvp(f, discretization="trigspec")[0]
        assert float((N(u) - f).norm(2)) < 1e3 * TOL    # pass(9)
        assert abs(float(u(DOM[0])) - float(u(DOM[1]))) < TOL  # pass(10)
        du = u.diff()
        assert abs(float(du(DOM[0])) - float(du(DOM[1]))) < TOL  # pass(11)
        assert isinstance(u.funs[0].tech, Trigtech)     # pass(12)
        assert u.isreal()                               # pass(13)
