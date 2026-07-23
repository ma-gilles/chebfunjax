"""Port of MATLAB Chebfun tests/chebfun2/test_max.m (Fable 5).

FIXED: Chebfun2.max2 added in the Fable 5 audit.

Provenance
----------
MATLAB source : tests/chebfun2/test_max.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2


class TestChebfun2Max:
    def test_max2(self):
        g = Chebfun2.from_function(
            lambda x, y: jnp.cos(2 * x) * jnp.cos(y))
        v, loc = g.max2()
        assert abs(float(v) - 1.0) < 1e-10


class TestChebfun2MaxDimensional:
    """MATLAB test_max.m: dimensional max returning 1D chebfuns."""

    TOL = 100 * float(np.finfo(np.float64).eps)

    def test_cos_xy(self):
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        t = jnp.asarray(np.linspace(-1.0, 1.0, 101))
        for h in (f.max(), f.max(None), f.max(None, 2)):
            assert float(np.max(np.abs(np.asarray(h(t)) - 1.0))) < self.TOL
        assert f.max(None, 3) is f

    def test_linear_rect_domain(self):
        f = Chebfun2.from_function(lambda x, y: x + 0 * y,
                                   domain=(-2.0, 3.0, -4.0, 10.0))
        tx = np.linspace(-2.0, 3.0, 101)
        ty = np.linspace(-4.0, 10.0, 101)
        for h in (f.max(), f.max(None)):
            assert float(np.max(np.abs(
                np.asarray(h(jnp.asarray(tx))) - tx))) < self.TOL
        h3 = f.max(None, 2)
        # MATLAB allows 1e12*tol here; we pass at the base tol.
        assert float(np.max(np.abs(
            np.asarray(h3(jnp.asarray(ty))) - 3.0))) < self.TOL
        assert f.max(None, 3) is f
