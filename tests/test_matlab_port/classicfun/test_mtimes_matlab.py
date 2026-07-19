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
        # pass(1): isempty(f*[]) && isempty([]*f) && isempty(2*g) && isempty(g*2)
        f = Bndfun.from_function(jnp.sin, DOM)
        e = Bndfun.empty()
        assert (f * e).isempty()
        assert (e * f).isempty()
        assert (2.0 * e).isempty()
        assert (e * 2.0).isempty()

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

    def test_array_valued_scalar_mult(self):
        # pass(5,6,7): scalar mtimes of an array-valued fun -- alpha*f == f*alpha,
        # values match, and 0*f is all-zero.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g1 = ALPHA * f
        g2 = f * ALPHA
        assert bool(jnp.all(jnp.asarray(g1(X)) == jnp.asarray(g2(X))))
        err = jnp.abs(jnp.asarray(g1(X)) - ALPHA * fop(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS
        assert bool(jnp.all(jnp.asarray((0 * f)(X)) == 0))

    def test_matrix_mtimes(self):
        # pass(8): f*A (matrix) -- MATLAB matrix mtimes.
        pytest.skip(
            "chebfunjax Bndfun has no matrix mtimes (f @ A); '*' is scalar/"
            "elementwise only"
        )

    def test_dimension_error(self):
        pytest.skip(
            "chebfunjax Bndfun has no matrix mtimes and no typed "
            "CHEBFUN:CLASSICFUN:mtimes:size error paths"
        )
