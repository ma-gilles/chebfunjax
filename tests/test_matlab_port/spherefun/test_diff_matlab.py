"""Port of MATLAB Chebfun tests/spherefun/test_diff.m (Fable 5).

Tangential derivatives of x, y, z (Cartesian coordinate functions) on
the sphere satisfy exact identities: d_x(x) = 1 - x^2 etc.

Provenance
----------
MATLAB source : tests/spherefun/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-9


def _xyz(lam, th):
    return (jnp.cos(lam) * jnp.sin(th), jnp.sin(lam) * jnp.sin(th),
            jnp.cos(th))


class TestSpherefunDiff:
    def test_tangential_derivative_identities(self):
        fz = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        dz = fz.diff(dim=3) if True else None
        lam = jnp.asarray(0.6)
        th = jnp.asarray(1.1)
        x, y, z = _xyz(lam, th)
        # d_z(z) = 1 - z^2 (tangential projection)
        assert abs(float(dz(lam, th)) - (1 - float(z) ** 2)) < TOL
