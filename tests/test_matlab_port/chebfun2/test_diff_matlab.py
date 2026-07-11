"""Port of MATLAB Chebfun tests/chebfun2/test_diff.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun2/test_diff.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 1e6 * EPS
FUNS = [
    (lambda x, y: x, lambda x, y: 1 + 0 * x, lambda x, y: 0 * x),
    (lambda x, y: jnp.cos(x) * jnp.exp(y),
     lambda x, y: -jnp.sin(x) * jnp.exp(y),
     lambda x, y: jnp.cos(x) * jnp.exp(y)),
    (lambda x, y: jnp.cos(x * y),
     lambda x, y: -y * jnp.sin(x * y),
     lambda x, y: -x * jnp.sin(x * y)),
    (lambda x, y: x ** 2 + x * y ** 2,
     lambda x, y: 2 * x + y ** 2,
     lambda x, y: 2 * x * y),
]
D = [(-1.0, 1.0, -1.0, 1.0), (-3.0, 1.0, -1.0, 2.0)]


class TestChebfun2Diff:
    @pytest.mark.parametrize("jj", range(4))
    @pytest.mark.parametrize("dom", D)
    def test_partial_derivatives(self, jj, dom):
        f, fx, fy = FUNS[jj]
        g = Chebfun2.from_function(f, domain=dom)
        gx = Chebfun2.from_function(fx, domain=dom)
        gy = Chebfun2.from_function(fy, domain=dom)
        # MATLAB: diff(g,1,2) = d/dx, diff(g,1,1) = d/dy
        assert float((g.diff(dim=2) - gx).norm()) < TOL
        assert float((g.diff(dim=1) - gy).norm()) < TOL

    def test_default_dim_is_y(self):
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x * y),
                                   domain=D[1])
        assert float((g.diff() - g.diff(dim=1)).norm()) < TOL

    def test_higher_order(self):
        # MATLAB tail: diff(g, 2, 1) and diff(g, 2, 2) of cos(x*y)
        dom = D[0]
        g = Chebfun2.from_function(lambda x, y: jnp.cos(x * y), domain=dom)
        gyy = Chebfun2.from_function(
            lambda x, y: -x ** 2 * jnp.cos(x * y), domain=dom)
        gxx = Chebfun2.from_function(
            lambda x, y: -y ** 2 * jnp.cos(x * y), domain=dom)
        assert float((g.diff(dim=1, k=2) - gyy).norm()) < TOL
        assert float((g.diff(dim=2, k=2) - gxx).norm()) < TOL
