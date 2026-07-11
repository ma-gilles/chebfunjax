"""Port of MATLAB Chebfun tests/chebfun/test_mtimes.m (Fable 5).

Scalar mtimes only; quasimatrix inner products via mtimes (f' * g) map
to innerProduct in chebfunjax.

Provenance
----------
MATLAB source : tests/chebfun/test_mtimes.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(6178)
X = jnp.asarray(2 * RNG.uniform(size=100) - 1)
ALPHA = -0.194758928283640 + 0.075474485412665j


class TestChebfunMtimes:
    def test_empty_cases(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_scalar_mtimes_kinked(self):
        funs = [_Piece.from_function(lambda x: -jnp.sin(x) * (x - 0.1),
                                     -1.0, 0.1),
                _Piece.from_function(lambda x: jnp.sin(x) * (x - 0.1),
                                     0.1, 1.0)]
        f1 = Chebfun(funs=funs, domain=Domain((-1.0, 0.1, 1.0)))
        g = ALPHA * f1
        exact = ALPHA * jnp.sin(X) * jnp.abs(X - 0.1)
        err = jnp.abs(g(X) - exact)
        assert float(jnp.max(err)) < 100 * f1.vscale * EPS

    def test_row_times_column_is_inner_product(self):
        # MATLAB: f' * g == innerProduct(f, g)
        f = cj.chebfun(jnp.sin, domain=(0.0, float(np.pi)))
        g = cj.chebfun(lambda x: x + 0.0)
        g = cj.chebfun(lambda x: x, domain=(0.0, float(np.pi)))
        ip = float(f.innerProduct(g))
        # exact: int_0^pi x sin x dx = pi
        assert abs(ip - np.pi) < 1e3 * EPS
