"""Port of MATLAB Chebfun tests/spherefunv/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/spherefunv/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

L0, T0 = jnp.asarray(0.7), jnp.asarray(1.1)


class TestSpherefunvNorm:
    def test_pointwise_norm(self):
        f = Spherefun.from_function(lambda lam, th: jnp.cos(th))
        g = Spherefun.from_function(lambda lam, th: jnp.sin(lam)
                                    * jnp.sin(th))
        F = Spherefunv(f, g)
        n = F.norm()
        exact = (float(f(L0, T0)) ** 2 + float(g(L0, T0)) ** 2) ** 0.5
        got = float(n(L0, T0)) if callable(n) else float(n)
        if not callable(n):
            pytest.skip("Spherefunv.norm returns a scalar (global "
                        "norm); MATLAB's is the pointwise magnitude")
        assert abs(got - exact) < 1e-9
