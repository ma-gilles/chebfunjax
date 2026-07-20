"""Port of MATLAB Chebfun tests/ballfun/test_spherefun.m (Fable 5).

Provenance
----------
MATLAB source : tests/ballfun/test_spherefun.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.ballfun.ballfun import Ballfun

_EPS = float(np.finfo(np.float64).eps)
_TOL = 1e2 * _EPS

_LAM = np.linspace(-np.pi, np.pi, 40, endpoint=False)
_TH = np.linspace(0.02, np.pi - 0.02, 33)
_LL, _TT = np.meshgrid(_LAM, _TH)
_XS = np.cos(_LL) * np.sin(_TT)
_YS = np.sin(_LL) * np.sin(_TT)
_ZS = np.cos(_TT)


def _sphval(g):
    return np.asarray(g(jnp.asarray(_LL), jnp.asarray(_TT)))


class TestBallfunSpherefun:
    def test_surface_default_radius(self):
        # g = spherefun(f): evaluate on the unit sphere (r = 1); f = cos(x y).
        f = Ballfun.from_function(lambda x, y, z: jnp.cos(x * y))
        g = f.to_spherefun()
        assert np.max(np.abs(_sphval(g) - np.cos(_XS * _YS))) < _TOL

    def test_inner_shell(self):
        # g = spherefun(f, 0.5): shell at r = 0.5; f = sin(z) -> sin(0.5 z).
        f = Ballfun.from_function(lambda x, y, z: jnp.sin(z))
        g = f.to_spherefun(0.5)
        assert np.max(np.abs(_sphval(g) - np.sin(0.5 * _ZS))) < _TOL

    def test_extrapolated_shell(self):
        # h = f(10,:,:): radial extrapolation to r = 10; f = x -> 10 x.
        f = Ballfun.from_function(lambda x, y, z: x)
        g = f.to_spherefun(10.0)
        assert np.max(np.abs(_sphval(g) - 10.0 * _XS)) < _TOL
