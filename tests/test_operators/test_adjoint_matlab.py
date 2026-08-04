"""MATLAB parity tests for chebop adjoint (ode-linear/Adjoints example).

Provenance
----------
MATLAB source : @chebop/adjoint.m, @linop/linopAdjoint.m
Chebfun commit: 7574c77
"""
import sys

import numpy as np

sys.path.insert(0, "src")

import chebfunjax as cj
from chebfunjax.operators.adjoint import adjoint
from chebfunjax.operators.chebop import Chebop


def _x():
    return cj.chebfun(lambda t: t, domain=(-1, 1))


class TestAdjointMatlab:
    def test_first_derivative(self):
        # L = diff, u(-1)=0  ->  L* = -diff, v(1)=0
        L = Chebop(lambda x, u: u.diff(), domain=(-1, 1))
        L.lbc = 0
        Ls = adjoint(L)
        assert Ls._disp_op_str == "-diff(v)"
        assert Ls._disp_lbc == []
        assert Ls._disp_rbc == ["v"]
        x = _x()
        u = (x + 1) * x.sin()
        v = (x - 1) * x.exp()
        lhs = float((v * u.diff()).sum())
        rhs = float((Ls.op(x, v) * u).sum())
        assert abs(lhs - rhs) < 1e-13

    def test_self_adjoint(self):
        L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
        L.lbc = 0
        L.rbc = 0
        Ls = adjoint(L)
        assert Ls._disp_op_str == "diff(v,2)+v"
        assert Ls._disp_lbc == ["v"]
        assert Ls._disp_rbc == ["v"]

    def test_ivp_becomes_fvp(self):
        L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
        L.lbc = [1, 0]
        Ls = adjoint(L)
        assert Ls._disp_lbc == []
        assert Ls._disp_rbc == ["v", "diff(v)"]

    def test_one_bc(self):
        L = Chebop(lambda x, u: u.diff(2) + u, domain=(-1, 1))
        L.lbc = 1
        Ls = adjoint(L)
        assert Ls._disp_lbc == ["v"]
        assert Ls._disp_rbc == ["v", "diff(v)"]

    def test_variable_coefficient(self):
        # L = x u'' -> L* = x v'' + 2 v'
        L = Chebop(lambda x, u: x * u.diff(2), domain=(-1, 1))
        L.lbc = 0
        L.rbc = 0
        Ls = adjoint(L)
        assert Ls._disp_op_str == "a11_2.*diff(v,2)+a11_1.*diff(v)"
        x = _x()
        # a2 = x, a1 = 2
        a2 = Ls._adj_coeffs[2]
        a1 = Ls._adj_coeffs[1]
        assert float((a2 - x).norm()) < 1e-12
        assert float((a1 - 2.0).norm()) < 1e-12
        u = (x**2 - 1) * x.sin()
        v = (x**2 - 1) * x.exp()
        lhs = float((v * (x * u.diff(2))).sum())
        rhs = float((Ls.op(x, v) * u).sum())
        assert abs(lhs - rhs) < 1e-13

    def test_biorthogonality(self):
        L = Chebop(lambda x, u: u.diff(2) - 20 * u.diff() + u,
                   domain=(-1, 1))
        L.lbc = 0
        L.rbc = 0
        Ls = adjoint(L)
        lam, V = L.eigs(k=3, sigma="SM", return_eigenfunctions=True)
        lams, Vs = Ls.eigs(k=3, sigma="SM",
                           return_eigenfunctions=True)
        lam = np.sort(np.real(np.asarray(lam)))
        lams = np.sort(np.real(np.asarray(lams)))
        # Adjoint spectrum equals the original (real operator)
        np.testing.assert_allclose(lam, lams, rtol=1e-6)
