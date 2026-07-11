"""Port of MATLAB Chebfun tests/chebfun/test_besselyk.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_besselyk.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import kv, yv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
XS = jnp.asarray(np.linspace(-0.95, 0.95, 60))


class TestChebfunBesselyk:
    def test_bessely_of_positive_function(self):
        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        h = f.bessely(1)
        exact = yv(1, 2 + np.sin(np.asarray(XS)))
        err = np.abs(np.asarray(h(XS)) - exact)
        assert float(np.max(err)) < 1e3 * EPS * max(h.vscale, 1.0)

    def test_besselk_of_positive_function(self):
        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        h = f.besselk(1)
        exact = kv(1, 2 + np.sin(np.asarray(XS)))
        err = np.abs(np.asarray(h(XS)) - exact)
        assert float(np.max(err)) < 1e3 * EPS * max(h.vscale, 1.0)
