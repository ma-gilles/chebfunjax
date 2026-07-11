"""Port of MATLAB Chebfun tests/chebfun2v/test_gradient.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2v/test_gradient.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

X0, Y0 = jnp.asarray(0.3), jnp.asarray(-0.4)


class TestChebfun2vGradient:
    def test_gradient_of_scalar_via_diff(self):
        # the scalar gradient (MATLAB gradient(chebfun2)) via diff:
        # Chebfun2.diff dim=2 -> d/dx, dim=1 -> d/dy
        f = Chebfun2.from_function(lambda x, y: x * x * y)
        gx = f.diff(dim=2)
        gy = f.diff(dim=1)
        assert abs(float(gx(X0, Y0)) - 2 * float(X0) * float(Y0)) < 1e-9
        assert abs(float(gy(X0, Y0)) - float(X0) ** 2) < 1e-9
