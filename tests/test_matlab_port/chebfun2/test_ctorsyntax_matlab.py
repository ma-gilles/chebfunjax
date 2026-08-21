"""Port of MATLAB Chebfun tests/chebfun2/test_ctorsyntax.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_ctorsyntax.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import chebfun2

jax.config.update("jax_enable_x64", True)


class TestChebfun2Ctorsyntax:
    def test_adaptive_calls(self):
        # pass(1): every syntax constructs without error and agrees.
        def f(x, y):
            return jnp.cos(x) + jnp.sin(x * y)

        g1 = chebfun2(f)
        g2 = chebfun2("cos(x) + sin(x.*y)")
        g3 = chebfun2(f, domain=(-1.0, 1.0, 1.0, 2.0))
        g4 = chebfun2(lambda x, y: x)
        xs = jnp.asarray(0.3)
        ys = jnp.asarray(0.4)
        want = float(np.cos(0.3) + np.sin(0.12))
        assert abs(float(g1(xs, ys)) - want) < 1e-13
        assert abs(float(g2(xs, ys)) - want) < 1e-13
        y2 = jnp.asarray(1.5)
        assert abs(float(g3(xs, y2))
                   - float(np.cos(0.3) + np.sin(0.45))) < 1e-13
        assert abs(float(g4(xs, ys)) - 0.3) < 1e-14

    def test_string_with_domain(self):
        g = chebfun2("sin(x.*y)", domain=(-2.0, 1.0, -2.0, 1.0))
        assert abs(float(g(jnp.asarray(-1.5), jnp.asarray(0.5)))
                   - float(np.sin(-0.75))) < 1e-13
