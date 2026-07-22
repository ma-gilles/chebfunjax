"""Port of MATLAB Chebfun tests/diskfunv/test_vertcat.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfunv/test_vertcat.m
Chebfun commit: 7574c77

MATLAB's ``[f; g]`` bracket syntax maps to ``Diskfun.vertcat(f, g)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp

from chebfunjax.diskfun.diskfun import Diskfun
from chebfunjax.diskfun.diskfunv import Diskfunv

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1e3 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunvVertcat:
    def test_vertcat_matches_diskfunv(self):
        f = _df(lambda t, r: jnp.cos(2 * jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t))))
        g = _df(lambda t, r: jnp.sin(2 * jnp.pi * (r * jnp.cos(t)) * (r * jnp.sin(t))))

        # pass(1)/(2): F = [f; g]; components recover f and g.
        F = Diskfun.vertcat(f, g)
        assert isinstance(F, Diskfunv)
        assert float((F.components[0] - f).norm()) < _TOL
        assert float((F.components[1] - g).norm()) < _TOL

        # pass(3)/(4): the direct Diskfunv constructor agrees.
        G = Diskfunv(f, g)
        assert float((G.components[0] - f).norm()) < _TOL
        assert float((G.components[1] - g).norm()) < _TOL

        # pass(5): norm(G - F) < tol (componentwise).
        for gc, fc in zip(G.components, F.components):
            assert float((gc - fc).norm()) < _TOL
