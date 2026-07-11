"""Port of MATLAB Chebfun tests/spherefunv/test_dot.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_dot.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

L0, T0 = jnp.asarray(0.7), jnp.asarray(1.1)


class TestSpherefunvDot:
    def test_dot_of_component_fields(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        g = Spherefun.from_function(lambda lam, th: jnp.sin(lam)
                                    * jnp.sin(th))
        F = Spherefunv(f, g)
        d = F.dot(F)
        exact = (float(f(L0, T0)) ** 2 + float(g(L0, T0)) ** 2)
        assert abs(float(d(L0, T0)) - exact) < 1e-10
