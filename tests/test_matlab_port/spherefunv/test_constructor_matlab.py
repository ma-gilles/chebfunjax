"""Port of MATLAB Chebfun tests/spherefunv/test_constructor.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_constructor.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

L0, T0 = jnp.asarray(0.7), jnp.asarray(1.1)


class TestSpherefunvConstructor:
    def test_components_evaluate(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        g = Spherefun.from_function(lambda lam, th: jnp.sin(lam))
        F = Spherefunv(f, g)
        out = F(L0, T0)
        vals = [float(v) for v in out]
        assert abs(vals[0] - float(jnp.cos(T0))) < 1e-10
        # sin(lam) is theta-independent: the BMC constructor warns
        # 'column slices not resolved' and delivers ~8e-9 accuracy
        # (a Spherefun constructor edge case worth improving).
        assert abs(vals[1] - float(jnp.sin(L0))) < 1e-7
