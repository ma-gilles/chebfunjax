"""Port of MATLAB Chebfun tests/ballfun/test_diskfun.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_diskfun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

_EPS = float(np.finfo(np.float64).eps)
_TOL = 1e3 * _EPS

# Sampling grid on the unit disk (theta, rho) with the disk area weight
# rho, so the discrete metric approximates MATLAB's L2 norm(g - h).
_TH = np.linspace(-np.pi, np.pi, 60, endpoint=False)
_RHO = np.linspace(0.0, 1.0, 50)
_THG, _RG = np.meshgrid(_TH, _RHO)
_XG = _RG * np.cos(_THG)
_YG = _RG * np.sin(_THG)


def _l2(diff):
    return float(np.sqrt(np.mean(diff**2 * _RG) / np.mean(_RG)))


def _diskval(g):
    return np.asarray(g(jnp.asarray(_THG), jnp.asarray(_RG)))


class TestBallfunDiskfun:
    def test_default_xy_slice(self):
        # g = diskfun(f) is the z=0 (xy-plane) slice; f = cos(x y).
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        g = f.to_diskfun()
        assert _l2(_diskval(g) - np.cos(_XG * _YG)) < _TOL

    def test_z_slice(self):
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        g = f.to_diskfun("z")
        assert _l2(_diskval(g) - np.cos(_XG * _YG)) < _TOL

    def test_y_slice(self):
        # f = x z; diskfun(f,'y') maps disk (x,y) -> ball (x, 0, z=y) = x y.
        f = Ballfun.from_function(lambda x, y, z: x * z)
        g = f.to_diskfun("y")
        assert _l2(_diskval(g) - _XG * _YG) < _TOL

    def test_z_slice_bilinear(self):
        f = Ballfun.from_function(lambda x, y, z: x * y)
        g = f.to_diskfun("z")
        assert _l2(_diskval(g) - _XG * _YG) < _TOL

    def test_offcenter_positive(self):
        # h = f(:,:,0.9): z = 0.9 plane, disk scaled by sqrt(1-0.9^2).
        f = Ballfun.from_function(
            lambda x, y, z: jnp.cos(z + np.pi * jnp.sin(np.pi * x)))
        g = f.to_diskfun("z", 0.9)
        exact = np.cos(0.9 + np.pi * np.sin(np.pi * _XG * np.sqrt(1 - 0.9**2)))
        assert _l2(_diskval(g) - exact) < _TOL

    def test_offcenter_negative(self):
        f = Ballfun.from_function(
            lambda x, y, z: jnp.cos(z + np.pi * jnp.sin(np.pi * x)))
        g = f.to_diskfun("z", -0.9)
        exact = np.cos(-0.9 + np.pi * np.sin(np.pi * _XG * np.sqrt(1 - 0.9**2)))
        assert _l2(_diskval(g) - exact) < _TOL
