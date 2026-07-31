"""Port of MATLAB Chebfun tests/chebop/test_zerothOrder.m (Fable 5).

Zeroth-order 'ODE': solve u^2 + sin(u) + exp(u) = sin(x) + 2 pointwise
via chebop Newton (MATLAB drives adchebfun manually; chebfunjax's
chebop.solve covers the same rootfind).

Provenance
----------
MATLAB source : tests/chebop/test_zerothOrder.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.operators.chebop import Chebop

TOL = 1e-8


class TestChebopZerothOrder:
    def test_pointwise_nonlinear_equation(self):
        # MATLAB pass(1): u^2 + sin(u) + exp(u) = sin(x) + 2, solved by
        # Newton; f is a chebfun built outside the op as in the MATLAB
        # source (the previous port called jnp.sin on the chebfun x and
        # skipped on the resulting TypeError -- a port bug, not a
        # chebop limitation).
        f = cj.chebfun(lambda t: jnp.sin(t) + 2)
        N = Chebop(lambda x, u: u * u + u.sin() + u.exp() - f)
        u = N.solve(0.0)
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 20))
        ux = np.asarray(u(xs))
        res = ux ** 2 + np.sin(ux) + np.exp(ux) \
            - (np.sin(np.asarray(xs)) + 2)
        assert float(np.max(np.abs(res))) < 1e-10
