"""Port of MATLAB Chebfun tests/chebfun/test_sign.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_sign.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunSign:
    def test_sign_of_positive_square(self):
        # sign(x^2) == 1 a.e. (test points avoid the root)
        f = cj.chebfun(lambda x: x ** 2 + 0.01)
        f1 = f.sign()
        assert bool(jnp.all(f1(X) == 1))

    def test_sign_of_shifted_cos(self):
        f = cj.chebfun(lambda x: jnp.cos(np.pi * x) + 2)
        f1 = f.sign()
        assert bool(jnp.all(f1(X) == 1))

    def test_point_values(self):
        # pass(1,2): sign propagates an explicit pointValues override
        # element-wise.  ``f = cos(pi x) + 2`` is strictly positive so sign
        # adds no breakpoints; the stored point values (all set to -pi) map to
        # sign(-pi) = -1.
        pts = np.sort(2 * RNG.uniform(size=8) - 1)
        f = cj.chebfun(lambda x: jnp.cos(np.pi * x) + 2,
                       domain=tuple(float(v) for v in pts))
        n_ends = len(f.domain.breakpoints)
        f = f.set_point_values(-np.pi * jnp.ones(n_ends))
        f1 = f.sign()
        assert bool(jnp.all(f1.point_values == -1))

    def test_sign_with_jump(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = cj.chebfun(lambda x: x - 0.3, splitting=True)
        f1 = f.sign()
        mask = jnp.abs(X - 0.3) > 1e-6
        exact = jnp.sign(X - 0.3)
        assert bool(jnp.all(f1(X)[mask] == exact[mask]))
