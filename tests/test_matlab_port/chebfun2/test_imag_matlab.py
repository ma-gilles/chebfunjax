"""Port of MATLAB Chebfun tests/chebfun2/test_imag.m (Fable 5).

FIXED (Fable 5 audit): ``Chebfun2.imag()`` now exists.

Provenance
----------
MATLAB source : tests/chebfun2/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun2Imag:
    def test_imag_consistency(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = Chebfun2.from_function(lambda x, y: jnp.sin(x + y ** 2))
        h = f + 1j * g
        # pass(1): imag(feval(h)) matches feval(g) on a 3x3 grid.
        xs = jnp.asarray(np.linspace(-1, 1, 3))
        xx, yy = jnp.meshgrid(xs, xs)
        assert float(jnp.max(jnp.abs(jnp.imag(h(xx, yy)) - g(xx, yy)))) < 10 * TOL
        # pass(2): imag(h) == g as a chebfun2.
        assert float((h.imag() - g).norm()) < TOL
