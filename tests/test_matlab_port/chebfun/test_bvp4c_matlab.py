"""Port of MATLAB Chebfun tests/chebfun/test_bvp4c.m (Fable 5).

MATLAB compares ``bvp4c`` started from a chebfun initial guess against
MATLAB's own ``bvp4c`` started from ``bvpinit``; here the reference is
``scipy.integrate.solve_bvp`` from the same coarse mesh and constant
guess.

Provenance
----------
MATLAB source : tests/chebfun/test_bvp4c.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_bvp

from chebfunjax.chebfun1d.chebfun import bvp4c, chebfun

jax.config.update("jax_enable_x64", True)


def _twoode(x, y):
    return jnp.array([y[1], -jnp.abs(y[0])])


def _twobc(ya, yb):
    return jnp.array([ya[0], yb[0] + 2.0])


class TestChebfunBvp4c:
    def test_all_matlab_assertions(self):
        d = (0.0, 4.0)
        y0 = chebfun(lambda x: jnp.stack([1 + 0 * x, 0 * x], axis=-1), domain=d)
        y = bvp4c(_twoode, _twobc, y0)
        x_mesh = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        sol = solve_bvp(lambda x, y: np.vstack([y[1], -np.abs(y[0])]),
                        lambda ya, yb: np.array([ya[0], yb[0] + 2.0]),
                        x_mesh, np.vstack([np.ones(5), np.zeros(5)]))
        assert np.max(np.abs(sol.y.T - np.asarray(y(jnp.asarray(sol.x))))) < 2e-2  # pass(1)

        tol = 1e-4
        p_true = 17.096591689705100
        sol1_true = np.array([-0.703689352093852, 1.033088348257337])
        q, lam = 5, 15.0
        mat4ode = lambda x, y, lam: jnp.array(  # noqa: E731
            [y[1], -(lam - 2 * q * jnp.cos(2 * x)) * y[0]])
        mat4bc = lambda ya, yb, lam: jnp.array([ya[1], yb[1], ya[0] - 1])  # noqa: E731
        mat4init = lambda x: jnp.stack([jnp.cos(4 * x), -4 * jnp.sin(4 * x)], axis=-1)  # noqa: E731
        opts = {"AbsTol": 1e-5, "RelTol": 1e-5}
        solinit = chebfun(mat4init, domain=(0.0, np.pi))
        sol, p = bvp4c(mat4ode, mat4bc, solinit, lam, opts)
        sol1 = np.asarray(sol(jnp.asarray(1.0)))
        assert np.max(np.abs(sol1 - sol1_true)) < tol               # pass(2)
        assert abs(float(p[0]) - p_true) < tol                      # pass(3)
