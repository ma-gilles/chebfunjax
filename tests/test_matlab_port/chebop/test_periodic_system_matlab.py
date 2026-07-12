"""Port of MATLAB Chebfun tests/chebop/test_periodic_system.m
(Fable 5).

FIXED: periodic systems of ODEs added in the Fable 5 audit (Fourier
collocation on an equispaced grid -- periodicity is built into the
trig representation, no boundary rows; Newton for the nonlinear
block).  Composing a trig piece (e.g. cos(v)) now stays in the
Fourier basis, which the residual arithmetic requires.

Provenance
----------
MATLAB source : tests/chebop/test_periodic_system.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import chebfun
from chebfunjax.operators.chebop import Chebop

DOM = (-np.pi, np.pi)
TOL = 2e-10
XS = jnp.asarray(np.linspace(-3.0, 3.0, 50))


class TestChebopPeriodicSystem:
    def test_nonlinear_periodic_system(self):
        # u - v' + v = 0, u'' - cos(v) = cos(x)
        N = Chebop(
            lambda x, u, v: [u - v.diff() + v,
                             u.diff(2) - v.cos()], DOM)
        N.bc = "periodic"
        fc = chebfun(jnp.cos, domain=DOM)
        z = chebfun(lambda x: 0 * x, domain=DOM)
        N.init = [z, fc]
        sol = N.solve([0, fc])
        u, v = sol[0], sol[1]
        r1 = float(jnp.max(jnp.abs(u(XS) - v.diff()(XS) + v(XS))))
        r2 = float(jnp.max(jnp.abs(
            u.diff(2)(XS) - jnp.cos(v(XS)) - jnp.cos(XS))))
        assert r1 < TOL
        assert r2 < TOL

    def test_linear_periodic_system(self):
        # u - v' = 0, u'' + v = cos(x)  (TRIGCOLLOC block)
        N = Chebop(
            lambda x, u, v: [u - v.diff(), u.diff(2) + v], DOM)
        N.bc = "periodic"
        fc = chebfun(jnp.cos, domain=DOM)
        sol = N.solve([0, fc])
        u, v = sol[0], sol[1]
        r1 = float(jnp.max(jnp.abs(u(XS) - v.diff()(XS))))
        r2 = float(jnp.max(jnp.abs(
            u.diff(2)(XS) + v(XS) - jnp.cos(XS))))
        assert r1 < TOL
        assert r2 < TOL
        # periodicity
        assert abs(float(u(jnp.asarray(DOM[0])))
                   - float(u(jnp.asarray(DOM[1] - 1e-12)))) < TOL
