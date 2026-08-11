"""Port of MATLAB Chebfun tests/chebfun2/test_scl.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): the scaling invariants are
checked directly through arithmetic and evaluation (``vscale`` exists on
Chebfun2, but the MATLAB test never calls it -- it checks that
construction is invariant under vertical and horizontal rescaling).

Provenance
----------
MATLAB source : tests/chebfun2/test_scl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun2d.chebfun2 import Chebfun2

EPS = float(np.finfo(np.float64).eps)
TOL = 10 * EPS


class TestChebfun2Scl:
    def test_vertical_scale_invariance(self):
        # pass(1): constructing eps*cos(x*y) directly agrees with
        # scaling cos(x*y) by eps.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        g = Chebfun2.from_function(lambda x, y: EPS * jnp.cos(x * y))
        assert float((EPS * f - g).norm()) < TOL

    def test_horizontal_scale_invariance(self):
        # pass(2, 3): cos(x*y) on [-1,1]^2 and cos((x/eps)(y/eps)) on
        # eps*[-1,1]^2 are the same function up to the rescaling.
        f = Chebfun2.from_function(lambda x, y: jnp.cos(x * y))
        dom = (-EPS, EPS, -EPS, EPS)
        g = Chebfun2.from_function(
            lambda x, y: jnp.cos((x / EPS) * (y / EPS)), domain=dom)
        assert abs(float(f(1.0, 1.0)) - float(g(EPS, EPS))) < TOL
        assert abs(float(f(np.pi / 6, 1.0))
                   - float(g(EPS * np.pi / 6, EPS))) < TOL

    def test_vscale_reports_vertical_scale(self):
        # vscale() reports the magnitude of the function.
        f = Chebfun2.from_function(lambda x, y: 3.0 * jnp.cos(x * y))
        assert abs(float(f.vscale()) - 3.0) < 1e3 * EPS
