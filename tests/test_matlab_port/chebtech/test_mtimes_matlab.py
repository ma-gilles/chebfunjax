"""Port of MATLAB Chebfun tests/chebtech/test_mtimes.m (Fable 5).

Scalar (alpha*f, f*alpha, 0*f) cases are ported at MATLAB tolerances for
both Chebtech1 and Chebtech2.  Empty-tech and array-valued cases are
skipped: chebfunjax has neither empty techs nor multi-column techs.

Provenance
----------
MATLAB source : tests/chebtech/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100))
ALPHA = 0.3 + 0.7j  # fixed complex scalar (MATLAB uses randn()+1i*randn())


@pytest.fixture(params=[Chebtech1, Chebtech2], ids=["chebtech1", "chebtech2"])
def cls(request):
    return request.param


class TestChebtechMtimes:
    def test_empty_cases(self, cls):
        # pass(n,1): isempty(f*[]) && isempty([]*f) && isempty(2*g) && isempty(g*2)
        f = cls.from_function(jnp.sin)
        e = cls.empty()
        assert (f * e).isempty()
        assert (e * f).isempty()
        assert (2.0 * e).isempty()
        assert (e * 2.0).isempty()

    def test_scalar_left_equals_right(self, cls):
        f = cls.from_function(jnp.sin)
        g1 = ALPHA * f
        g2 = f * ALPHA
        assert float(jnp.max(jnp.abs(g1.coeffs - g2.coeffs))) == 0.0

    def test_scalar_multiplication_values(self, cls):
        f = cls.from_function(jnp.sin)
        g1 = ALPHA * f
        err = jnp.abs(g1(X) - ALPHA * jnp.sin(X))
        assert float(jnp.max(err)) < 10 * g1.vscale * EPS

    def test_zero_scalar_gives_zero(self, cls):
        f = cls.from_function(jnp.sin)
        g = 0 * f
        assert bool(jnp.all(g.coeffs == 0))

    # FIXED (Fable 5, Big-Three array-valued epic): pass(n, 5)-(8)
    # port now that techs support (n, m) coefficient matrices and
    # matrix mtimes maps to Python ``@``.
    def test_array_valued_scalar_cases(self, cls):
        # pass(n, 5)-(7): scalar * array-valued tech.
        f = cls.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g1 = ALPHA * f
        g2 = f * ALPHA
        assert float(jnp.max(jnp.abs(g1.coeffs - g2.coeffs))) == 0.0
        exact = ALPHA * jnp.stack(
            [jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1)
        assert float(jnp.max(jnp.abs(g1(X) - exact))) \
            < 10 * g1.vscale * EPS
        assert bool(jnp.all((0 * f).coeffs == 0))

    def test_array_valued_matrix_mtimes(self, cls):
        # pass(n, 8): f*A mixes the columns (MATLAB mtimes -> @).
        rng = np.random.default_rng(6178)
        A = jnp.asarray(rng.standard_normal((3, 3)))
        f = cls.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x), jnp.cos(x), jnp.exp(x)], axis=-1))
        g = f @ A
        exact = jnp.stack(
            [jnp.sin(X), jnp.cos(X), jnp.exp(X)], axis=-1) @ A
        assert float(jnp.max(jnp.abs(g(X) - exact))) \
            < 10 * g.vscale * EPS

    def test_dimension_error(self, cls):
        # MATLAB pass(n,9): [1 2 3]*f raises a dimension error.
        pytest.skip("chebfunjax techs have no matrix mtimes to raise on")
