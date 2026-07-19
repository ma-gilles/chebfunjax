"""Port of MATLAB Chebfun tests/chebfun/test_gmres.m (Fable 5).

FIXED (Fable 5): chebfun-level gmres(L, f) solves the linear operator
equation L(u) = f by GMRES with Chebfun inner products.

Provenance
----------
MATLAB source : tests/chebfun/test_gmres.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, gmres

TOL = float(np.finfo(np.float64).eps)


class TestChebfunGmres:
    def test_all_matlab_assertions(self):
        d = (-1.0, 1.0)
        f = Chebfun.from_function(lambda t: jnp.exp(t), Domain(d))
        w = 100.0

        def L(u):
            return u.diff() + 1j * w * u

        u, flag = gmres(L, f)
        # pass(1): converged.
        assert flag == 0

        # pass(2): the solution satisfies the integration-by-parts identity
        # <f, exp(1i w x)> == u(1) exp(1i w) - u(-1) exp(-1i w).
        e = Chebfun.from_function(lambda t: jnp.exp(1j * w * t), Domain(d))
        lhs = complex((f * e).sum())
        rhs = (complex(u(jnp.asarray(1.0))) * np.exp(1j * w)
               - complex(u(jnp.asarray(-1.0))) * np.exp(-1j * w))
        assert abs(lhs - rhs) < 100 * TOL
