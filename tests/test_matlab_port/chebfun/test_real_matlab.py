"""Port of MATLAB Chebfun tests/chebfun/test_real.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_real.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
XS = jnp.asarray(np.linspace(-0.95, 0.95, 60))


class TestChebfunReal:
    def test_real_of_complex_exponential(self):
        f = cj.chebfun(lambda x: jnp.exp(1j * np.pi * x))
        g = f.real()
        err = jnp.abs(g(XS) - jnp.cos(np.pi * XS))
        assert float(jnp.max(err)) < 100 * EPS

    def test_real_of_real_is_identity(self):
        f = cj.chebfun(jnp.sin)
        g = f.real()
        err = jnp.abs(g(XS) - f(XS))
        assert float(jnp.max(err)) < 10 * EPS
