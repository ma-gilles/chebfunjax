"""Port of MATLAB Chebfun tests/linop/test_discretization.m (Fable 5).

Uses the chebcolloc2 discretization, as the MATLAB test does.

Provenance
----------
MATLAB source : tests/linop/test_discretization.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.blocks import ChebColloc2Disc, D, I, mult
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.utils.quadrature import chebpts_ab

jax.config.update("jax_enable_x64", True)

D55 = np.array([
    [-2.750000000000000, 3.414213562373094, -1.000000000000000,
     0.585786437626905, -0.250000000000000],
    [-0.853553390593274, 0.353553390593274, 0.707106781186548,
     -0.353553390593274, 0.146446609406726],
    [0.250000000000000, -0.707106781186548, 0.0,
     0.707106781186548, -0.250000000000000],
    [-0.146446609406726, 0.353553390593274, -0.707106781186548,
     -0.353553390593274, 0.853553390593274],
    [0.250000000000000, -0.585786437626905, 1.000000000000000,
     -3.414213562373094, 2.750000000000000],
])


class TestLinopDiscretization:
    def test_all_matlab_assertions(self):
        dom = (-2.0, 2.0)
        Id = ChebMatrix([[I(dom)]])
        Dop = ChebMatrix([[D(dom)]])
        u = cj.chebfun(lambda x: x ** 2, domain=dom)
        U = ChebMatrix([[mult(u)]])

        err = []
        err.append(float(jnp.linalg.norm(Id.dense(5) - jnp.eye(5))))
        err.append(float(jnp.linalg.norm(Dop.dense(5) - D55)))
        xx = chebpts_ab(5, dom[0], dom[1], kind=2)
        err.append(float(jnp.linalg.norm(U.dense(5) - jnp.diag(u(xx)))))

        dom = (-2.0, 1.0, 1.5, 2.0)
        Id = ChebMatrix([[I(dom)]])
        u = cj.chebfun(lambda x: x ** 2, domain=dom)
        U = ChebMatrix([[mult(u)]])
        n = [5, 5, 5]

        err.append(float(jnp.linalg.norm(Id.dense(n) - jnp.eye(sum(n)))))
        xx = ChebColloc2Disc(n, dom).points()
        err.append(float(jnp.linalg.norm(U.dense(n) - jnp.diag(u(xx)))))

        assert all(e < 1e-9 for e in err), err
