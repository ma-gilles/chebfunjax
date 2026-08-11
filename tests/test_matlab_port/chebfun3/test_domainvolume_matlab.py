"""Port of MATLAB Chebfun tests/chebfun3/test_domainvolume.m (Fable 5).

FIXED (Fable 5, chebfun2/3 skip sweep): ``Chebfun3.domainvolume()``
now exists.  MATLAB pass(3) builds a 'trig' Chebfun3; the trigonometric
tech option is not wired into the constructor, so that case is covered
here on the same box without the flag (domainvolume depends only on the
domain).

Provenance
----------
MATLAB source : tests/chebfun3/test_domainvolume.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d.chebfun3 import chebfun3

EPS = float(np.finfo(np.float64).eps)
TOL = 1000 * EPS


def _ff(x, y, z):
    return jnp.cos(x + y + z)


class TestChebfun3Domainvolume:
    def test_default_domain(self):
        # pass(1): [-1,1]^3 has volume 8.
        assert chebfun3(_ff).domainvolume() == 8.0

    def test_shifted_domain(self):
        # pass(2): [-1,2]x[-2,1]x[-3,0] has volume 27.
        f = chebfun3(_ff, domain=(-1.0, 2.0, -2.0, 1.0, -3.0, 0.0))
        assert f.domainvolume() == 27.0

    def test_periodic_box(self):
        # pass(3): [-pi,pi]^3 has volume (2 pi)^3.
        dom = (-np.pi, np.pi, -np.pi, np.pi, -np.pi, np.pi)
        f = chebfun3(_ff, domain=dom)
        assert abs(f.domainvolume() - (2 * np.pi) ** 3) < TOL
