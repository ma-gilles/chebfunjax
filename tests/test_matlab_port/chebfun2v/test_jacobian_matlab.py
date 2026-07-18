"""Port of MATLAB Chebfun tests/chebfun2v/test_jacobian.m (Fable 5).

FIXED (Fable 5): Chebfun2v.jacobian() implemented
(F1_x F2_y - F1_y F2_x, MATLAB @chebfun2v/jacobian.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_jacobian.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

EPS = float(np.finfo(np.float64).eps)


class TestChebfun2vJacobian:
    def test_definition(self):
        # pass(1): jacobian(F) == F1_x.*F2_y - F1_y.*F2_x,
        # tol 100*cheb2eps.  F = [cos(x); sin(y)] gives
        # jac = -sin(x) cos(y).
        f1 = Chebfun2.from_function(lambda x, y: jnp.cos(x))
        f2 = Chebfun2.from_function(lambda x, y: jnp.sin(y))
        F = Chebfun2v([f1.approx, f2.approx])
        jac = F.jacobian()
        exact = Chebfun2.from_function(
            lambda x, y: -jnp.sin(x) * jnp.cos(y))
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 8))
        err = float(jnp.max(jnp.abs(jac(xs, xs) - exact(xs, xs))))
        assert err < 100 * EPS
