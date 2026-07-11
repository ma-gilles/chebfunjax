"""Port of MATLAB Chebfun tests/chebfun/test_besselj.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_besselj.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import jv

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
XR = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunBesselj:
    def test_besselj_of_exp(self):
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        h = f.besselj(2)
        exact = jnp.asarray(jv(2, np.exp(np.asarray(XR))))
        err = jnp.abs(h(XR) - exact)
        assert float(jnp.max(err)) < 100 * EPS * max(h.vscale, 1.0)

    def test_scale_option(self):
        pytest.skip("chebfunjax besselj has no MATLAB scale option")
