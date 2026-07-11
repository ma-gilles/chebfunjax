"""Port of MATLAB Chebfun tests/chebfun2v/test_cross.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_cross.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vCross:
    def test_scalar_cross_2d(self):
        # cross((f,g),(h,k)) = f*k - g*h
        F = Chebfun2v.from_functions(lambda x, y: x, lambda x, y: y)
        G = Chebfun2v.from_functions(lambda x, y: -y, lambda x, y: x)
        c = F.cross(G)
        exact = float(X0) ** 2 + float(Y0) ** 2
        assert abs(float(c(X0, Y0)) - exact) < 1e-10
