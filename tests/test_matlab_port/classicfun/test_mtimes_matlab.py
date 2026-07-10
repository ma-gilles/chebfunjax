"""Port of MATLAB Chebfun tests/classicfun/test_mtimes.m (Fable 5).

Scalar cases at MATLAB tolerances on the MATLAB domain [-2, 7];
empty-fun and array-valued cases are skipped (chebfunjax has neither).

Provenance
----------
MATLAB source : tests/classicfun/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))
ALPHA = 0.3 + 0.7j


class TestClassicfunMtimes:
    def test_empty_cases(self):
        pytest.skip("chebfunjax has no empty Bndfun representation")

    def test_scalar_left_equals_right(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA * f
        g2 = f * ALPHA
        err = jnp.abs(jnp.asarray(g1(X)) - jnp.asarray(g2(X)))
        assert float(jnp.max(err)) == 0.0

    def test_scalar_multiplication_values(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g1 = ALPHA * f
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * jnp.sin(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS

    def test_zero_scalar_gives_zero(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g = 0 * f
        assert bool(jnp.all(jnp.asarray(g(X)) == 0))

    def test_array_valued_cases(self):
        pytest.skip("chebfunjax has no array-valued Bndfun")

    def test_dimension_error(self):
        pytest.skip("chebfunjax funs have no matrix mtimes to raise on")
