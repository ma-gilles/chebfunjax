"""Port of MATLAB Chebfun tests/chebfun/test_besselh.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_besselh.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.special import hankel1

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunBesselh:
    def test_hankel_of_positive_function(self):
        f = cj.chebfun(lambda x: 2 + jnp.sin(x))
        # chebfunjax besselh returns (real_part, imag_part) chebfuns
        hre, him = f.besselh(1)
        xs = jnp.asarray(np.linspace(-0.95, 0.95, 60))
        got = (np.asarray(hre(xs)) + 1j * np.asarray(him(xs)))
        exact = hankel1(1, 2 + np.sin(np.asarray(xs)))
        err = np.abs(got - exact)
        assert float(np.max(err)) < 1e3 * EPS
