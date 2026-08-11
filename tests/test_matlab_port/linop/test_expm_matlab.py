"""Port of MATLAB Chebfun tests/linop/test_expm.m (Fable 5).

Ports the chebcolloc2 blocks (MATLAB err(1,:), err(2,:), err(7,:), err(8,:)).
The chebcolloc1 and ultraS repeats are covered by a separate skipped test.

Provenance
----------
MATLAB source : tests/linop/test_expm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

import chebfunjax as cj
from chebfunjax.operators.blocklinop import linop
from chebfunjax.operators.blocks import (
    primitive_functionals,
    primitive_operators,
)
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

TOL = 1e-9


def _at(f, x):
    return float(f(jnp.asarray(float(x))))


class TestLinopExpm:
    def test_all_matlab_assertions(self):
        d = (-math.pi, math.pi)
        x = cj.chebfun(lambda t: t, domain=d)
        Z, I, Dop, C, M = primitive_operators(d)
        z, E, s, dt = primitive_functionals(d)

        A = linop(Dop ** 2)
        A = A.add_constraint(E(-math.pi), 0.0)
        A = A.add_constraint(E(math.pi), 0.0)

        # Smooth initial condition.
        u0 = (x.exp()).sin() * (math.pi ** 2 - x ** 2)
        u = A.expm(0.02, u0)
        exact = -4.720369127510475
        err = [abs(_at(u[0], math.pi / 2) - exact),
               abs(_at(u[0], -math.pi)),
               abs(_at(u[0], math.pi))]
        assert all(e < TOL for e in err), err

        # Piecewise initial condition.
        u0 = cj.chebfun(lambda t: -abs(t) / math.pi + 1,
                        domain=(-math.pi, 0.0, math.pi))
        u = A.expm(0.01, u0)
        exact = 0.95545945604534127  # mathematica
        err = [abs(_at(u[0], 0.1) - exact),
               abs(_at(u[0], -math.pi)),
               abs(_at(u[0], math.pi))]
        assert all(e < TOL for e in err), err

        # t = 0 must reproduce the initial condition exactly, at its length.
        A = linop(Dop ** 2 + Dop)
        A = A.add_constraint(E(-math.pi), 0.0)
        A = A.add_constraint(E(math.pi), 0.0)
        u0 = (-55 * x ** 2).exp()
        v = A.expm(0, u0)
        assert float(abs((u0 - v[0]).norm())) < TOL
        assert len(v[0]) == len(u0)

        # t = 0 for a 2x2 system.
        d = (-1.0, 1.0)
        Z, I, Dop, C, M = primitive_operators(d)
        z, E, s, dt = primitive_functionals(d)
        x = cj.chebfun(lambda t: t, domain=d)
        U0 = [(2 * math.pi * x).sin() ** 2, 1 + 0 * x]
        A = linop(ChebMatrix([[Dop ** 2, Z], [Z, Dop ** 2]]))
        A = A.add_constraint([E(-1.0), z], 0.0)
        A = A.add_constraint([E(1.0), z], 0.0)
        A = A.add_constraint([z, E(-1.0)], 0.0)
        A = A.add_constraint([z, E(1.0)], 0.0)
        V = A.expm(0, U0)
        assert float(abs((U0[0] - V[0]).norm())) < TOL
        assert float(abs((U0[1] - V[1]).norm())) < TOL
        assert [len(V[0]), len(V[1])] == [len(U0[0]), len(U0[1])]

    @pytest.mark.skip(
        reason="MATLAB err(3,:) through err(6,:) repeat the first two "
               "propagations with the chebcolloc1 and ultraS "
               "discretizations; chebfunjax's BlockLinop only implements "
               "chebcolloc2 rectangular collocation.")
    def test_ultras_and_chebcolloc1(self):
        raise NotImplementedError
