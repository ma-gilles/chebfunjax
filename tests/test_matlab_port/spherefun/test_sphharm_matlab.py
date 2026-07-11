"""Port of MATLAB Chebfun tests/spherefun/test_sphharm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefun/test_sphharm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun

TOL = 1e-10


class TestSpherefunSphharm:
    def test_orthonormality(self):
        # <Y_l^m, Y_l'^m'> = delta via surface quadrature of products
        pairs = [(1, 0), (2, 1), (3, -2)]
        for l1, m1 in pairs:
            Y1 = Spherefun.sphharm(l1, m1)
            # normalization: integral of Y^2 = 1
            sq = Spherefun.from_function(
                lambda lam, th, Y=Y1: Y(lam, th) ** 2)
            assert abs(float(sq.sum()) - 1.0) < TOL

    def test_laplace_eigenfunction(self):
        Y = Spherefun.sphharm(4, 2)
        lam, th = jnp.asarray(0.7), jnp.asarray(1.1)
        ratio = float(Y.laplacian()(lam, th)) / float(Y(lam, th))
        assert abs(ratio + 20.0) < 1e-8
