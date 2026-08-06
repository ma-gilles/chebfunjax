"""Higher-order and complex IVP systems marched by reduction to first order.

MATLAB reduces such a problem with ``treeVar.toFirstOrder`` and hands it
to ``ode113``. ``_solve_ivp_system`` only handled systems that are first
order in every unknown -- it recovers the right-hand side by evaluating
the operator on CONSTANT chebfuns, which makes every derivative vanish
-- so a second-order system fell through to collocation and ground to a
halt (ode-nonlin/ThreePlanets did not finish in 900s on a fifth of its
interval). ``_solve_ivp_system_highorder`` carries the derivative tower
in the state instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from chebfunjax.operators.chebop import Chebop


class TestRealHigherOrderSystem:
    def test_second_order_pair_against_exact_solution(self):
        # u'' = -u, v'' = -4v with u(0)=0, u'(0)=1, v(0)=0, v'(0)=2
        N = Chebop(lambda t, u, v: [u.diff(2) + u, v.diff(2) + 4 * v],
                   domain=(0, 2))
        N.lbc = lambda u, v: [u, u.diff() - 1, v, v.diff() - 2]
        u, v = N.solve(0.0)
        t = np.linspace(0, 2, 64)
        assert np.asarray(u(t)) == pytest.approx(np.sin(t), abs=1e-9)
        assert np.asarray(v(t)) == pytest.approx(np.sin(2 * t), abs=1e-8)

    def test_mixed_orders_in_one_system(self):
        # w' = u, u'' = -u  =>  u = sin t, w = 1 - cos t
        N = Chebop(lambda t, u, w: [u.diff(2) + u, w.diff() - u],
                   domain=(0, 3))
        N.lbc = lambda u, w: [u, u.diff() - 1, w]
        u, w = N.solve(0.0)
        t = np.linspace(0, 3, 64)
        assert np.asarray(u(t)) == pytest.approx(np.sin(t), abs=1e-8)
        assert np.asarray(w(t)) == pytest.approx(1 - np.cos(t), abs=1e-8)

    def test_initial_conditions_are_hit_exactly(self):
        N = Chebop(lambda t, u, v: [u.diff(2) + u, v.diff(2) + 4 * v],
                   domain=(0, 2))
        N.lbc = lambda u, v: [u - 0.25, u.diff() - 1, v + 0.5, v.diff() - 2]
        u, v = N.solve(0.0)
        assert float(u(np.float64(0.0))) == pytest.approx(0.25, abs=1e-11)
        assert float(v(np.float64(0.0))) == pytest.approx(-0.5, abs=1e-11)


class TestComplexHigherOrderSystem:
    """The state must stay complex: a float() cast silently drops Im."""

    @staticmethod
    def _planets(domain):
        def planetfun(t, x, y, z):
            fYX = (y - x) / abs(y - x) ** 3
            fZX = (z - x) / abs(z - x) ** 3
            fZY = (z - y) / abs(z - y) ** 3
            return [x.diff(2) - fYX - fZX,
                    y.diff(2) + fYX - fZY,
                    z.diff(2) + fZX + fZY]
        N = Chebop(planetfun, domain=domain)
        N.lbc = lambda x, y, z: [x, y - 3, z - 4j,
                                 x.diff(), y.diff(), z.diff()]
        return N

    def test_complex_initial_state_is_preserved(self):
        x, y, z = self._planets((0, 4)).solve(0.0)
        assert complex(x(np.float64(0.0))) == pytest.approx(0j, abs=1e-11)
        assert complex(y(np.float64(0.0))) == pytest.approx(3 + 0j, abs=1e-11)
        # the imaginary part of z(0) is the whole point
        assert complex(z(np.float64(0.0))) == pytest.approx(4j, abs=1e-11)

    def test_centre_of_mass_is_conserved(self):
        # Equal masses starting at rest: the centre of mass cannot move.
        x, y, z = self._planets((0, 8)).solve(0.0)

        def com(t):
            return (complex(x(np.float64(t))) + complex(y(np.float64(t)))
                    + complex(z(np.float64(t)))) / 3.0

        c0 = com(0.0)
        assert c0 == pytest.approx((3 + 4j) / 3.0, abs=1e-12)
        for t in (2.0, 5.0, 8.0):
            assert com(t) == pytest.approx(c0, abs=1e-9)

    def test_total_momentum_stays_zero(self):
        x, y, z = self._planets((0, 8)).solve(0.0)
        p = x.diff() + y.diff() + z.diff()
        for t in (0.0, 3.0, 8.0):
            assert complex(p(np.float64(t))) == pytest.approx(0j, abs=1e-8)


class TestRoutingIsUnchanged:
    def test_first_order_systems_still_take_the_original_path(self):
        # Lotka-Volterra: first order in both unknowns, so the
        # higher-order marcher must not be involved.
        N = Chebop(lambda t, u, v: [u.diff() - u + u * v,
                                    v.diff() + v - u * v], domain=(0, 4))
        N.lbc = lambda u, v: [u - 0.5, v - 1]
        u, v = N.solve(0.0)
        assert float(u(np.float64(0.0))) == pytest.approx(0.5, abs=1e-10)
        assert float(v(np.float64(0.0))) == pytest.approx(1.0, abs=1e-10)

    def test_scalar_second_order_ivp_is_unaffected(self):
        # u'' + u = 0, u(0)=0, u'(0)=1 -> sin t, via the scalar path.
        N = Chebop(lambda t, u: u.diff(2) + u, domain=(0, 2))
        N.lbc = [0.0, 1.0]
        u = N.solve(0.0)
        t = np.linspace(0, 2, 32)
        assert np.asarray(u(t)) == pytest.approx(np.sin(t), abs=1e-8)


class TestScalarComplexIVP:
    """A scalar complex IVP routes through the system marcher (m = 1).

    ``_solve_ivp`` works in float64, so ode-nonlin/TwoElectrons -- which
    writes the plane as a single complex z -- raised out of it after
    doing full adaptive construction first.
    """

    def test_complex_initial_value_is_exact(self):
        # z'' = -z with z(0) = 1i, z'(0) = 1  =>  z = i cos t + sin t
        N = Chebop(lambda t, z: z.diff(2) + z, domain=(0, 3))
        N.lbc = [1j, 1.0]
        z = N.solve(0.0)
        assert complex(z(np.float64(0.0))) == pytest.approx(1j, abs=1e-11)
        t = np.linspace(0, 3, 48)
        want = 1j * np.cos(t) + np.sin(t)
        assert np.asarray(z(t)) == pytest.approx(want, abs=1e-8)

    def test_list_bc_on_one_unknown_means_successive_derivatives(self):
        # The scalar convention: entry j is the j-th derivative. The
        # system convention (one value per unknown) must NOT be applied
        # here -- conflating them silently sets the wrong initial state.
        N = Chebop(lambda t, z: z.diff(2) + z, domain=(0, 2))
        N.lbc = [2.0, -1.0]                     # z(0) = 2, z'(0) = -1
        z = N.solve(0.0)
        assert float(z(np.float64(0.0))) == pytest.approx(2.0, abs=1e-11)
        assert float(z.diff()(np.float64(0.0))) == pytest.approx(
            -1.0, abs=1e-8)

    def test_complex_scalar_is_not_silently_realified(self):
        # A float() cast anywhere on this path would drop Im and leave a
        # plausible-looking real trajectory.
        N = Chebop(lambda t, z: z.diff(2) + z, domain=(0, 2))
        N.lbc = [1j, 0.0]
        z = N.solve(0.0)
        vals = np.asarray(z(np.linspace(0, 2, 16)))
        assert np.max(np.abs(vals.imag)) > 0.5
        assert np.max(np.abs(vals.real)) < 1e-8
