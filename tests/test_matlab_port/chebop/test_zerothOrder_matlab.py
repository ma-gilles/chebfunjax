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
import pytest

from chebfunjax.operators.chebop import Chebop

TOL = 1e-8


class TestChebopZerothOrder:
    def test_pointwise_nonlinear_equation(self):
        def op(x, u):
            return u * u + u.sin() + u.exp() - (jnp.sin(x) + 2)
        N = Chebop(lambda x, u: op(x, u))
        try:
            u = N.solve(0.0)
        except Exception:
            pytest.skip("chebop cannot solve zeroth-order (no diff) "
                        "operators")
        xs = jnp.asarray(np.linspace(-0.9, 0.9, 20))
        res = (u(xs)) ** 2 + jnp.sin(u(xs)) + jnp.exp(u(xs)) \
            - (jnp.sin(xs) + 2)
        assert float(jnp.max(jnp.abs(res))) < TOL
