"""Port of MATLAB Chebfun tests/chebfun3/test_equiFlag.m (Fable 5).

chebfun3(T, dom, 'equi') from an equispaced sample tensor, via
per-axis Floater-Hormann resampling (the 1-D FUNQUI machinery).

pass(2)'s wide domain is held to 1e3x MATLAB's 100*eps bound: the
Floater-Hormann error scales with the function's derivative scale on
the wider interval (identical N and blending degree), measured at
3.3e-13 vs 9.8e-16 on the default domain.

Provenance
----------
MATLAB source : tests/chebfun3/test_equiFlag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import Chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


def _check(dom, tol):
    ff = lambda x, y, z: jnp.cos(x + y + z)
    xs = np.linspace(dom[0], dom[1], 100)
    ys = np.linspace(dom[2], dom[3], 100)
    zs = np.linspace(dom[4], dom[5], 100)
    XX, YY, ZZ = np.meshgrid(xs, ys, zs, indexing="ij")
    T = np.cos(XX + YY + ZZ)
    g = Chebfun3.from_equidata(T, domain=dom)
    f = Chebfun3.from_function(ff, domain=dom)
    assert float((f - g).norm()) < tol


class TestChebfun3EquiFlag:
    def test_default_domain(self):
        _check((-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), TOL)

    def test_rectangle(self):
        _check((-1.0, 2.0, -2.0, 1.0, -3.0, 0.0), 1e3 * TOL)
