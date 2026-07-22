"""Port of MATLAB Chebfun tests/diskfun/test_power.m (Fable 5).

Provenance
----------
MATLAB source : tests/diskfun/test_power.m
Chebfun commit: 7574c77

Cartesian ``@(x,y)`` handles from MATLAB are written directly in polar
coordinates ``(theta, r)`` with ``x = r cos(theta)``, ``y = r sin(theta)``.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

from chebfunjax.diskfun.diskfun import Diskfun

_EPS = float(jnp.finfo(jnp.float64).eps)
_TOL = 1000 * _EPS


def _df(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Diskfun.from_function(fn)


class TestDiskfunPower:
    def test_square_via_sample(self):
        # pass(1): f = x, g = x^2; norm(sample(f,100,100)^2 - sample(g,100,100))
        f = _df(lambda t, r: r * jnp.cos(t))
        g = _df(lambda t, r: (r * jnp.cos(t)) ** 2)
        sf = np.asarray(f.sample(100, 100))
        sg = np.asarray(g.sample(100, 100))
        assert np.linalg.norm(sf**2 - sg, np.inf) < _TOL

    def test_diskfun_to_diskfun_power(self):
        # pass(2): f = cos(x*y), g = cos(x*y)^cos(x*y); norm(sample(f^f)-sample(g))
        def xy(t, r):
            return (r * jnp.cos(t)) * (r * jnp.sin(t))

        f = _df(lambda t, r: jnp.cos(xy(t, r)))
        g = _df(lambda t, r: jnp.cos(xy(t, r)) ** jnp.cos(xy(t, r)))
        fpf = f**f
        sf = np.asarray(fpf.sample(100, 100))
        sg = np.asarray(g.sample(100, 100))
        assert np.linalg.norm(sf - sg) < _TOL
