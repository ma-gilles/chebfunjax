"""Scalar polynomial partial-derivative controls via diff.

These are supplemental Python checks, not translated native test assertions.
The formerly cited native test file does not exist at Chebfun7574c77.
The corresponding library operation is documented in @separableApprox/diff.m;
the analytic formulas and diagnostic bounds below are supplemental oracles.
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
