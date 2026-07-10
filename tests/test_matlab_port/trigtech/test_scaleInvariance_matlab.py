"""Port of MATLAB Chebfun tests/trigtech/test_scaleInvariance.m (Opus 4.8).

Vertical-scale invariance of construction: scaling the sampled function by
a constant scales the Fourier coefficients by exactly the same constant.

Provenance
----------
MATLAB source : tests/trigtech/test_scaleInvariance.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

from chebfunjax.tech.trigtech import Trigtech


def _F(x):
    return jnp.sin(10 * jnp.pi * x)


def _tt(f):
    return Trigtech.from_function(f)


def _anynonzero(a):
    return bool(jnp.any(jnp.abs(jnp.asarray(a)) > 0))


class TestTrigtechScaleInvariance:
    def test_scale_up(self):
        # scale * TRIGTECH(f) == TRIGTECH(scale * f)
        f = _tt(_F)
        scale = 2.0**300
        f1 = _tt(lambda x: _F(x) * scale)
        assert f.n == f1.n
        assert not _anynonzero(f.coeffs - f1.coeffs / scale)

    def test_scale_down(self):
        # TRIGTECH(f) / scale == TRIGTECH(f / scale)
        f = _tt(_F)
        scale = 2.0**300
        f2 = _tt(lambda x: _F(x) / scale)
        assert f.n == f2.n
        assert not _anynonzero(f.coeffs - f2.coeffs * scale)
