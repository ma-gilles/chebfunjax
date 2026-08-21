"""Port of MATLAB Chebfun tests/chebfun3/test_subsref.m (Fable 5).

MATLAB's colon cross-sections map to restrict() with degenerate
intervals (1 fixed coordinate -> Chebfun2).

Provenance
----------
MATLAB source : tests/chebfun3/test_subsref.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

jax.config.update("jax_enable_x64", True)

TOL = 1e4 * 2.220446049250313e-16
DOM = (-2.0, 2.0, -3.0, 4.0, -np.pi, np.pi)


def _ff(x, y, z):
    return jnp.sin(x * (y - 0.1) * (z + 0.2))


def _grid2(g, ref, ax, bx, ay, by):
    xs = jnp.linspace(ax + 1e-9, bx - 1e-9, 9)
    ys = jnp.linspace(ay + 1e-9, by - 1e-9, 9)
    X, Y = jnp.meshgrid(xs, ys)
    return float(jnp.max(jnp.abs(jnp.asarray(g(X, Y))
                                 - ref(X, Y))))


class TestChebfun3Subsref:
    def test_all_matlab_assertions(self):
        f = Chebfun3.from_function(_ff, domain=DOM)

        # pass(1): point evaluation.
        p = (np.pi / 4, np.pi / 6, np.pi / 3)
        assert abs(float(f(*map(jnp.asarray, p)))
                   - float(_ff(*map(jnp.asarray, p)))) < TOL

        # pass(2): f(:, :, z0) cross-sections.
        for z0 in (np.pi / 6, np.pi / 4):
            g = f.restrict((DOM[0], DOM[1], DOM[2], DOM[3], z0, z0))
            assert _grid2(
                g, lambda X, Y, _z=z0: _ff(X, Y, jnp.asarray(_z)),
                DOM[0], DOM[1], DOM[2], DOM[3]) < TOL

        # pass(3)/(4): f(x0, :, :) cross-sections.
        for x0 in (np.pi / 4, np.pi / 6):
            g = f.restrict((x0, x0, DOM[2], DOM[3], DOM[4], DOM[5]))
            assert _grid2(
                g, lambda Y, Z, _x=x0: _ff(jnp.asarray(_x), Y, Z),
                DOM[2], DOM[3], DOM[4], DOM[5]) < TOL

        # pass(5): the full restriction is f itself.
        g = f.restrict(DOM)
        xs = jnp.linspace(-1.9, 1.9, 5)
        ys = jnp.linspace(-2.9, 3.9, 5)
        zs = jnp.linspace(-3.0, 3.0, 5)
        X, Y, Z = jnp.meshgrid(xs, ys, zs)
        assert float(jnp.max(jnp.abs(
            jnp.asarray(g(X, Y, Z)) - jnp.asarray(f(X, Y, Z))))) < TOL
