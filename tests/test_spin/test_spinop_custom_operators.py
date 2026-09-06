"""Custom (user-defined) operators for spinop2/spinop3 (Fable 5).

MATLAB defines a 2-D/3-D SPINOP by assigning anonymous functions,
``S = spinop2(dom, tspan); S.lin = @(u,v) [...]; S.nonlin = @(u,v) [...];
S.init = ...``; chebfunjax takes the same MATLAB strings (and Python
callables for the nonlinear part / initial condition).

Provenance
----------
MATLAB source : @spinop2/spinop2.m, @spinop3/spinop3.m,
    @spinoperator/discretize.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.spinop2 import Spinop2, spin2
from chebfunjax.operators.spinop3 import Spinop3, spin3
from chebfunjax.operators.spinop_parse import parse_lin_handle
from chebfunjax.operators.spinopsphere import func2str

jax.config.update("jax_enable_x64", True)


class TestParse:
    def test_lin_handle_coefficients(self):
        names, coeffs = parse_lin_handle("@(u,v) [2e-5*lap(u); 1e-5*lap(v)]")
        assert names == ("u", "v")
        assert coeffs == [(2e-5, 0.0, 0.0, 0.0, 0.0), (1e-5, 0.0, 0.0, 0.0, 0.0)]
        _, c = parse_lin_handle("@(u) -2*lap(u) - biharm(u)")
        assert c == [(-2.0, -1.0, 0.0, 0.0, 0.0)]
        _, c = parse_lin_handle("@(u) (1+1i)*lap(u) + triharm(u)")
        assert c[0][0] == (1 + 1j) and c[0][2] == 1.0

    def test_lin_handle_rejects_off_diagonal_and_unknown_terms(self):
        with pytest.raises(ValueError):
            parse_lin_handle("@(u,v) [lap(v); lap(u)]")
        with pytest.raises(ValueError):
            parse_lin_handle("@(u) diff(u, 2)")


class TestSpinop2Custom:
    def test_custom_gray_scott_matches_preset(self):
        F, K = 0.030, 0.057
        S = Spinop2((0.0, 1.0, 0.0, 1.0), (0.0, 0.1))
        S.lin = "@(u,v) [2e-5*lap(u); 1e-5*lap(v)]"
        S.nonlin = ("@(u,v) [%g*(1 - u) - u.*v.^2; -(%g+%g)*v + u.*v.^2]"
                    % (F, F, K))
        assert func2str(S.lin) == "@(u,v)[2e-5*lap(u);1e-5*lap(v)]"
        assert S.numVars == 2
        P = Spinop2("GS")
        S.init = P.init
        P.tspan = (0.0, 0.1)
        u = spin2(S, 32, 1e-2, "dealias", "off")
        v = spin2(P, 32, 1e-2, "dealias", "off")
        g = jnp.linspace(0.0, 1.0, 17)
        XX, YY = jnp.meshgrid(g, g)
        for c in range(2):
            a = np.asarray(u.components[c](XX, YY))
            b = np.asarray(v.components[c](XX, YY))
            assert np.max(np.abs(a - b)) < 1e-13

    def test_custom_heat_equation_exact(self):
        # u_t = lap(u), u0 = cos(x) cos(y) on [0, 2pi]^2 -> exp(-2t) u0;
        # ETDRK4 is exact for a linear problem.
        S = Spinop2((0.0, 2 * np.pi, 0.0, 2 * np.pi), (0.0, 0.5))
        S.lin = "@(u) lap(u)"
        S.nonlin = "@(u) 0*u"
        S.init = lambda x, y: jnp.cos(x) * jnp.cos(y)
        u = spin2(S, 16, 1e-2, "dealias", "off")
        g = jnp.linspace(0.0, 2 * np.pi, 9)
        XX, YY = jnp.meshgrid(g, g)
        exact = np.exp(-2 * 0.5) * np.cos(np.asarray(XX)) * np.cos(np.asarray(YY))
        assert np.max(np.abs(np.asarray(u(XX, YY)) - exact)) < 1e-12

    def test_python_callable_nonlinearity(self):
        S = Spinop2((0.0, 2 * np.pi, 0.0, 2 * np.pi), (0.0, 0.1))
        S.lin = "@(u) lap(u)"
        S.nonlin = lambda u: -u ** 3
        S.init = lambda x, y: 0.5 * jnp.cos(x) * jnp.cos(y)
        T = Spinop2((0.0, 2 * np.pi, 0.0, 2 * np.pi), (0.0, 0.1))
        T.lin = "@(u) lap(u)"
        T.nonlin = "@(u) -u.^3"
        T.init = S.init
        u = spin2(S, 16, 1e-2, "dealias", "off")
        v = spin2(T, 16, 1e-2, "dealias", "off")
        g = jnp.linspace(0.0, 2 * np.pi, 9)
        XX, YY = jnp.meshgrid(g, g)
        assert np.max(np.abs(np.asarray(u(XX, YY)) - np.asarray(v(XX, YY)))) < 1e-14


class TestSpinop3Custom:
    def test_custom_heat_equation_exact(self):
        S = Spinop3((0.0, 2 * np.pi, 0.0, 2 * np.pi, 0.0, 2 * np.pi), (0.0, 0.2))
        S.lin = "@(u) lap(u)"
        S.nonlin = "@(u) 0*u"
        S.init = lambda x, y, z: jnp.cos(x) * jnp.cos(y) * jnp.cos(z)
        u = spin3(S, 8, 1e-2, "dealias", "off")
        g = jnp.linspace(0.0, 2 * np.pi, 5)
        XX, YY, ZZ = jnp.meshgrid(g, g, g)
        exact = (np.exp(-3 * 0.2) * np.cos(np.asarray(XX)) * np.cos(np.asarray(YY))
                 * np.cos(np.asarray(ZZ)))
        assert np.max(np.abs(np.asarray(u(XX, YY, ZZ)) - exact)) < 1e-12

    def test_custom_schnakenberg_matches_preset(self):
        S = Spinop3((0.0, 25.0, 0.0, 25.0, 0.0, 25.0), (0.0, 5e-3))
        S.lin = "@(u,v) [lap(u); 10*lap(v)]"
        S.nonlin = "@(u,v) [3*(.1 - u + u.^2.*v); 3*(.9 - u.^2.*v)]"
        P = Spinop3("SCHNAK")
        S.init = P.init
        P.tspan = (0.0, 5e-3)
        u = spin3(S, 8, 1e-3, "dealias", "off")
        v = spin3(P, 8, 1e-3, "dealias", "off")
        g = jnp.linspace(0.0, 25.0, 5)
        XX, YY, ZZ = jnp.meshgrid(g, g, g)
        for c in range(2):
            a = np.asarray(u.components[c](XX, YY, ZZ))
            b = np.asarray(v.components[c](XX, YY, ZZ))
            assert np.max(np.abs(a - b)) < 1e-13
