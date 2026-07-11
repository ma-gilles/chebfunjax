"""Port of MATLAB Chebfun tests/chebfun/test_times.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_times.m
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


def _kinked():
    funs = [_Piece.from_function(lambda x: -jnp.sin(x) * (x - 0.1),
                                 -1.0, 0.1),
            _Piece.from_function(lambda x: jnp.sin(x) * (x - 0.1),
                                 0.1, 1.0)]
    return Chebfun(funs=funs, domain=Domain((-1.0, 0.1, 1.0)))


def _f1_op(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


class TestChebfunTimes:
    def test_empty_cases(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_scalar_multiply(self):
        f1 = _kinked()
        g1 = f1 * ALPHA
        g2 = ALPHA * f1
        exact = _f1_op(X) * ALPHA
        assert float(jnp.max(jnp.abs(g1(X) - exact))) \
            < 100 * f1.vscale * EPS
        assert float(jnp.max(jnp.abs(g2(X) - g1(X)))) == 0.0

    def test_multiply_two_functions(self):
        f1 = _kinked()
        g = cj.chebfun(lambda x: jnp.cos(x) * jnp.exp(x))
        h = f1 * g
        exact = _f1_op(X) * jnp.cos(X) * jnp.exp(X)
        assert float(jnp.max(jnp.abs(h(X) - exact))) \
            < 1e3 * max(f1.vscale, g.vscale) * EPS

    def test_square_is_nonnegative(self):
        f1 = _kinked()
        h = f1 * f1
        assert bool(jnp.all(h(X) >= -100 * EPS))
