"""Port of MATLAB Chebfun tests/chebfun/test_sqrt.m (Fable 5).

Positive-function square roots on [-2, 7] at MATLAB tolerances;
root-touching/singular cases skipped (need blowup at the chebfun level).

Provenance
----------
MATLAB source : tests/chebfun/test_sqrt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(np.sort(9 * RNG.uniform(size=100) - 2))


class TestChebfunSqrt:
    def test_runge_reciprocal(self):
        f = cj.chebfun(lambda x: 1.0 / (1 + 25 * x ** 2),
                       domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = 1.0 / jnp.sqrt(1 + 25 * X ** 2)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e2 * EPS * float(jnp.max(exact))

    def test_oscillatory_positive(self):
        f = cj.chebfun(lambda x: jnp.sin(50 * x) ** 2 + 1,
                       domain=(-2.0, 7.0))
        g = f.sqrt()
        exact = jnp.sqrt(jnp.sin(50 * X) ** 2 + 1)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 1e3 * EPS * float(jnp.max(exact))

    def test_root_touching(self):
        # sqrt of a non-negative function that touches zero at its endpoints
        # carries a branch point into a fractional (0.5) SingFun exponent
        # (MATLAB @chebfun/power.m -> @singfun/power.m).  1 - x^2 vanishes at
        # x = +/-1, so sqrt(1 - x^2) has exponents (0.5, 0.5).
        f = cj.chebfun(lambda x: 1.0 - x ** 2, domain=(-1.0, 1.0))
        g = f.sqrt()
        from chebfunjax.fun.singfun import Singfun
        assert isinstance(g.funs[0].tech, Singfun)
        assert g.funs[0].tech.exponents == (0.5, 0.5)
        assert not bool(g.isinf())
        xt = jnp.asarray(np.linspace(-0.98, 0.98, 50))
        exact = jnp.sqrt(1.0 - xt ** 2)
        err = float(jnp.max(jnp.abs(g(xt) - exact)))
        assert err < 1e2 * EPS

    def test_root_touching_interior(self):
        # A non-negative function with an INTERIOR double root: sqrt gets a
        # breakpoint at the root and stays finite (|x| on either side).
        f = cj.chebfun(lambda x: x ** 2, domain=(-1.0, 1.0))
        g = f.sqrt()
        xt = jnp.asarray(np.linspace(-0.97, 0.97, 41))
        err = float(jnp.max(jnp.abs(g(xt) - jnp.abs(xt))))
        assert err < 1e2 * EPS
        assert not bool(g.isinf())
