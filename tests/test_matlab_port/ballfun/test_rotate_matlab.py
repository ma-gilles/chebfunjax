"""Port of MATLAB Chebfun tests/ballfun/test_rotate.m (Fable 5).

FIXED: Ballfun.rotate added in the Fable 5 audit (ZYZ Euler).

Provenance
----------
MATLAB source : tests/ballfun/test_rotate.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

TOL = 1e3 * np.finfo(float).eps
RS = jnp.asarray(np.linspace(0.05, 1.0, 7))
LAMS = jnp.asarray(np.linspace(-3, 3, 7))
THS = jnp.asarray(np.linspace(0.1, 3.0, 7))
RR, LL, TT = jnp.meshgrid(RS, LAMS, THS, indexing="ij")


def _nrm(u, ex):
    return float(jnp.max(jnp.abs(u(RR, LL, TT) - ex(RR, LL, TT))))


class TestBallfunRotate:
    def test_all_matlab_assertions(self):
        f = Ballfun.from_function(
            lambda r, lam, th: r * jnp.sin(th) * jnp.cos(lam),
            spherical=True)
        # pass(1)
        assert _nrm(f.rotate(np.pi, 0, 0),
                    lambda r, lam, th: r * jnp.sin(th)
                    * jnp.cos(lam - np.pi)) < TOL
        # pass(2)
        assert _nrm(f.rotate(np.pi / 2, 0, 0),
                    lambda r, lam, th: r * jnp.sin(th)
                    * jnp.cos(lam - np.pi / 2)) < TOL
        # pass(3)
        fz = Ballfun.from_function(
            lambda r, lam, th: r * jnp.cos(th), spherical=True)
        assert _nrm(fz.rotate(0, np.pi, 0),
                    lambda r, lam, th: -r * jnp.cos(th)) < TOL
        # pass(4)
        fy = Ballfun.from_function(lambda x, y, z: y)
        g = fy.rotate(-np.pi, 0, np.pi)
        assert _nrm(g, lambda r, lam, th:
                    r * jnp.sin(th) * jnp.sin(lam)) < TOL
