"""Port of MATLAB Chebfun tests/chebfun/test_abs.m (Fable 5).

The pointValues case has no counterpart (no pointValues field).  The
array-valued abs case is ported (per-column ``|.|`` with union-of-columns
breakpoints).

Provenance
----------
MATLAB source : tests/chebfun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)


class TestChebfunAbs:
    def test_abs_of_square_piece_count(self):
        f = cj.chebfun(lambda x: x ** 2)
        f1 = f.abs()
        assert len(f1.funs) in (1, 2)

    def test_abs_nonnegative(self):
        f = cj.chebfun(lambda x: jnp.cos(3 * x))
        f1 = f.abs()
        assert bool(jnp.all(f1(X) >= 0))

    def test_abs_values_match(self):
        f = cj.chebfun(lambda x: jnp.cos(3 * x))
        f1 = f.abs()
        err = jnp.abs(f1(X) - jnp.abs(jnp.cos(3 * X)))
        assert float(jnp.max(err)) < 10 * f.vscale * EPS

    def test_point_values(self):
        pytest.skip("chebfunjax Chebfun has no pointValues field")

    def test_array_valued(self):
        # pass(5): abs of an array-valued chebfun (4 pieces on -2:2, per-column
        # |.|, nonnegative).  FIXED (Fable 5, Big-Three array-valued epic).
        X3 = jnp.asarray(4 * RNG.uniform(size=100) - 2)
        g = cj.chebfun(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * (x - 0.5))], axis=-1
            ),
            domain=(-2, -1, 0, 1, 2),
        )
        h = g.abs()
        assert len(h.funs) == 4
        # MATLAB tol 20*eps*hscale, hscale = 2 (half-width of [-2, 2]).
        assert float(jnp.max(jnp.abs(h(X3) - jnp.abs(g(X3))))) < 20 * EPS * 2
        assert bool(jnp.all(h(X3) >= 0))
