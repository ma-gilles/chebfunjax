"""Port of MATLAB Chebfun tests/chebfun2/test_diag.m (Fable 5).

FIXED: Chebfun2.diag_fun added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_diag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Diag:
    def test_diagonal_chebfun(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y) + x)
        d = f.diag_fun()
        for t in (0.3, -0.6):
            assert abs(float(d(jnp.asarray(t)))
                       - float(f(jnp.asarray(t), jnp.asarray(t)))) \
                < 1e-13
