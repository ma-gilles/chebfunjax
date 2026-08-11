"""Port of MATLAB Chebfun tests/chebfun3/test_scl.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.vscale()`` now
exists; the MATLAB test itself checks that construction is invariant
under vertical and horizontal rescaling.

Provenance
----------
MATLAB source : tests/chebfun3/test_scl.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 100 * EPS


class TestChebfun3Scl:
    def test_vertical_scale_invariance(self):
        # pass(1): eps*f built directly equals eps times f.
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
        g = chebfun3(lambda x, y, z: EPS * jnp.cos(x * y * z))
        assert float((EPS * f - g).norm()) < TOL

    def test_horizontal_scale_invariance(self):
        # pass(2, 3): the same function on the eps-scaled box.
        f = chebfun3(lambda x, y, z: jnp.cos(x * y * z))
        dom = (-EPS, EPS, -EPS, EPS, -EPS, EPS)
        g = chebfun3(
            lambda x, y, z: jnp.cos((x / EPS) * (y / EPS) * (z / EPS)),
            domain=dom)
        assert abs(float(f(1.0, 1.0, 1.0))
                   - float(g(EPS, EPS, EPS))) < TOL
        assert abs(float(f(np.pi / 6, 1.0, 1.0))
                   - float(g(EPS * np.pi / 6, EPS, EPS))) < TOL

    def test_vscale_reports_vertical_scale(self):
        # vscale() reports the magnitude of the function.
        f = chebfun3(lambda x, y, z: 3.0 * jnp.cos(x * y * z))
        assert abs(f.vscale() - 3.0) < 1e3 * EPS
