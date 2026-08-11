"""Port of MATLAB Chebfun tests/linop/test_fitBCs.m (Fable 5).

Ports the chebcolloc2 pass (MATLAB k = 2).  MATLAB's fitBCs fixes the
discretization to chebcolloc2 regardless of the preference, so the three
passes of the MATLAB loop are identical; the loop is therefore run once.

Provenance
----------
MATLAB source : tests/linop/test_fitBCs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    jump_at,
    jump_functional,
    primitive_functionals,
    primitive_operators,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-10


def _at(f, x, side=None):
    return float(f(jnp.asarray(float(x)), side) if side
                 else f(jnp.asarray(float(x))))


class TestLinopFitBCs:
    def test_all_matlab_assertions(self):
        Z, I, Dop, C, M = primitive_operators()
        zr, ev, su, dt = primitive_functionals()
        err = []

        # Scalar equation.
        L = linop(Dop ** 2)
        L = L.add_constraint(su, -2.0)
        L = L.add_constraint(ev(-1.0) + ev(1.0), -3.0)
        u = L.fit_bcs()
        err.append(float(np.linalg.norm([
            float(u[0].sum()) - 2.0,
            _at(u[0], -1.0) + _at(u[0], 1.0) - 3.0])))

        # Scalar equation (higher-order constraints).
        L = linop(Dop ** 3)
        L = L.add_constraint(su, -2.0)
        L = L.add_constraint(ev(-1.0) + ev(1.0), -3.0)
        L = L.add_constraint(ev(0.0) * Dop ** 2, 0.0)
        u = L.fit_bcs()
        err.append(float(np.linalg.norm([
            float(u[0].sum()) - 2.0,
            _at(u[0], -1.0) + _at(u[0], 1.0) - 3.0,
            _at(u[0].diff(2), 0.0)])))

        # System.
        L = linop(ChebMatrix([[Dop ** 2, I], [Dop ** 2, -I]]))
        L = L.add_constraint([su, zr], -1.0)
        L = L.add_constraint([ev(0.0), -ev(1.0)], 0.0)
        L = L.add_constraint([zr, ev(0.5) - ev(0.7)], 0.0)
        L = L.add_constraint([zr, su], -2.0)
        uu = L.fit_bcs()
        err.append(float(np.linalg.norm([
            float(uu[0].sum()) - 1.0,
            _at(uu[0], 0.0) - _at(uu[1], 1.0),
            _at(uu[1], 0.5) - _at(uu[1], 0.7),
            float(uu[1].sum()) - 2.0])))

        # System with jumps.
        dom = (0.0, 0.3, 1.0)
        Z, I, Dop, C, M = primitive_operators(dom)
        z, e, s, dt = primitive_functionals(dom)
        j = jump_at(dom)
        A = linop(ChebMatrix([[Dop ** 2, I], [-Dop, Dop ** 2 + I]]))
        A = A.add_constraint([e(0.0), z], -1.0)
        A = A.add_constraint([e(1.0), z], 0.0)
        A = A.add_constraint([z, e(0.0)], 0.0)
        A = A.add_constraint([z, e(1.0)], 1.0)
        A = A.add_continuity([j(0.3, 1), z], 2.0)
        A = A.add_continuity([z, j(0.3, 1)], 0.0)
        A = A.add_continuity([j(0.3, 0), z], 0.0)
        A = A.add_continuity([z, j(0.3, 0)], -0.5)
        uu = A.fit_bcs()
        u1, u2 = uu[0], uu[1]
        J = jump_functional(0.3, dom, 0)
        err.append(float(np.linalg.norm([
            _at(u1, 0.0) - 1.0,
            _at(u1, 1.0),
            _at(u2, 0.0),
            _at(u2, 1.0) + 1.0,
            float(J * u1.diff()) + 2.0,
            float(J * u1),
            float(J * u2) - 0.5,
            float(J * u2.diff())])))

        assert all(v < TOL for v in err), err
