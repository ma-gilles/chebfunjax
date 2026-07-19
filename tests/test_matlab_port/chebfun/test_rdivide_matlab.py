"""Port of MATLAB Chebfun tests/chebfun/test_rdivide.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunRdivide:
    def test_empty_cases(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.sin(x))
        g = chebfun()
        assert (f / g).isempty()
        assert (g / f).isempty()

    def test_zero_over_f_is_zero(self):
        f = cj.chebfun(jnp.sin)
        g = 0.0 / (f + 2)          # avoid roots of sin
        assert float(jnp.max(jnp.abs(g(X)))) == 0.0

    def test_reciprocal_of_exp(self):
        f = cj.chebfun(jnp.exp, domain=[-1.0, -0.5, 0.0, 0.5, 1.0])
        g = 1.0 / f
        err = jnp.abs(g(X) - jnp.exp(-X))
        assert float(jnp.max(err)) < 100 * EPS * float(np.e)

    def test_function_over_function(self):
        f = cj.chebfun(jnp.sin)
        g = cj.chebfun(lambda x: 2 + jnp.cos(x))
        h = f / g
        exact = jnp.sin(X) / (2 + jnp.cos(X))
        err = jnp.abs(h(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS
