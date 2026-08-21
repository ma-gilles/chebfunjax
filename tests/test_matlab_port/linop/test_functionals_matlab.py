"""Port of MATLAB Chebfun tests/linop/test_functionals.m (Fable 5).

Provenance
----------
MATLAB source : tests/linop/test_functionals.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import chebfunjax as cj
from chebfunjax.operators.blocks import primitive_functionals
from chebfunjax.operators.chebmatrix import ChebMatrix

jax.config.update("jax_enable_x64", True)

EPS = 2.220446049250313e-16


class TestLinopFunctionals:
    def test_all_matlab_assertions(self):
        d = (-1.0, 2.0)
        x = cj.chebfun(lambda t: t, domain=d)
        f = x.cos() / (1 + x ** 2)

        z, e, s, dt = primitive_functionals(d)

        A = ChebMatrix([[s - z], [-2 * dt(x ** 2)]])
        Af = A * ChebMatrix([[f]])
        assert abs(float(f.sum()) - float(Af[0, 0])) < 100 * EPS
        assert abs(-2 * float(f.inner(x ** 2)) - float(Af[1, 0])) < 100 * EPS

        A = ChebMatrix([[e(2.0)], [e(0.0)]])
        Af = A * ChebMatrix([[f]])
        assert abs(float(f(jnp.asarray(2.0))) - float(Af[0, 0])) < 100 * EPS
        assert abs(float(f(jnp.asarray(0.0))) - float(Af[1, 0])) < 100 * EPS
