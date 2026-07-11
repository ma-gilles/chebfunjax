"""Port of MATLAB Chebfun tests/chebfun/test_ellipke.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_ellipke.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ellipk

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


class TestChebfunEllipke:
    def test_ellipke_of_identity(self):
        m = cj.chebfun(lambda x: x, domain=(0.0, 0.99))
        K1 = m.ellipke()
        K1 = K1[0] if isinstance(K1, tuple) else K1
        xs = jnp.asarray(np.linspace(0.01, 0.98, 60))
        exact = jnp.asarray(ellipk(np.asarray(xs)))
        err = jnp.abs(K1(xs) - exact)
        assert float(jnp.max(err)) < 1e2 * EPS * K1.vscale

    def test_array_valued(self):
        pytest.skip("chebfunjax has no array-valued chebfun")
