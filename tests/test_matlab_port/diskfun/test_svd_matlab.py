"""Port of MATLAB Chebfun tests/diskfun/test_svd.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_svd.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS  # 1000 * chebfun2eps-scale tolerance (as in MATLAB test)


def _cart(fn):
    # MATLAB constructs f = diskfun(@(x,y) ...) in Cartesian coordinates;
    # chebfunjax from_function takes polar (theta, r): x = r cos t, y = r sin t.
    return Diskfun.from_function(
        lambda t, r: fn(r * jnp.cos(t), r * jnp.sin(t))
    )


class TestDiskfunSvd:
    def test_norm_matches_singular_values(self):
        f = _cart(lambda x, y: jnp.cos(x * y**2))
        s = np.asarray(f.svd())
        # ||f||^2 == sum(s.^2)
        assert abs(float(f.norm()) ** 2 - float(np.sum(s**2))) < _TOL

    def test_resolved_tail(self):
        f = _cart(lambda x, y: jnp.cos(x * y**2))
        s = np.asarray(f.svd())
        assert s[-1] < 1e1 * _TOL

    def test_scale_invariant(self):
        f = _cart(lambda x, y: jnp.cos(x * y**2))
        s = np.asarray(f.svd())
        g = 100.0 * f
        t = np.asarray(g.svd())
        assert np.linalg.norm(s - t / 100.0) < _TOL
