"""Port of MATLAB Chebfun tests/chebfun2v/test_divgrad.m (Fable 5).

FIXED (Fable 5): Chebfun2v.divgrad() implemented
(d^2 F1/dx^2 + d^2 F2/dy^2, MATLAB @chebfun2v/divgrad.m).

Provenance
----------
MATLAB source : tests/chebfun2v/test_divgrad.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebfun2d.chebfun2v import Chebfun2v

CHEB2EPS = float(np.finfo(np.float64).eps)


class TestChebfun2vDivgrad:
    def test_definition(self):
        # pass(1): divgrad(F) == diffx(F1, 2) + diffy(F2, 2),
        # tol = 50 * cheb2eps.
        f1 = Chebfun2.from_function(lambda x, y: jnp.cos(x))
        f2 = Chebfun2.from_function(lambda x, y: jnp.sin(y))
        F = Chebfun2v([f1.approx, f2.approx])
        dg = F.divgrad()
        exact = f1.diff(2, 2) + f2.diff(1, 2)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 8))
        ys = jnp.asarray(np.linspace(-0.9, 0.9, 8))
        err = float(jnp.max(jnp.abs(dg(xs, ys) - exact(xs, ys))))
        assert err < 50 * CHEB2EPS
