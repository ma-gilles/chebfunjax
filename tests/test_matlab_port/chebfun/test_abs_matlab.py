"""Port of MATLAB Chebfun tests/chebfun/test_abs.m (Fable 5).

Array-valued and pointValues cases are skipped (no counterparts).

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
        pytest.skip("chebfunjax has no array-valued Chebfun")
