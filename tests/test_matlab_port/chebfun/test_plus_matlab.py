"""Port of MATLAB Chebfun tests/chebfun/test_plus.m (Fable 5).

Scalar-add and function-add on smooth and piecewise (kink) chebfuns at
MATLAB tolerances.  Empty/array-valued/singular-exponent cases skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_plus.m
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
    # f1 = sin(x)*|x - 0.1| built piecewise (MATLAB splitting result)
    funs = [_Piece.from_function(lambda x: -jnp.sin(x) * (x - 0.1),
                                 -1.0, 0.1),
            _Piece.from_function(lambda x: jnp.sin(x) * (x - 0.1),
                                 0.1, 1.0)]
    return Chebfun(funs=funs, domain=Domain((-1.0, 0.1, 1.0)))


def _f1_op(x):
    return jnp.sin(x) * jnp.abs(x - 0.1)


class TestChebfunPlus:
    def test_empty_cases(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_add_scalar_both_sides(self):
        f1 = _kinked()
        g1 = f1 + ALPHA
        g2 = ALPHA + f1
        exact = _f1_op(X) + ALPHA
        assert float(jnp.max(jnp.abs(g1(X) - exact))) \
            < 10 * f1.vscale * EPS * 10
        assert float(jnp.max(jnp.abs(g2(X) - g1(X)))) == 0.0

    def test_add_two_functions(self):
        f1 = _kinked()
        g = cj.chebfun(lambda x: jnp.cos(x) * jnp.exp(x))
        h = f1 + g
        exact = _f1_op(X) + jnp.cos(X) * jnp.exp(X)
        assert float(jnp.max(jnp.abs(h(X) - exact))) \
            < 100 * max(f1.vscale, g.vscale) * EPS

    def test_commutativity(self):
        f1 = _kinked()
        g = cj.chebfun(lambda x: jnp.cos(x))
        d = (f1 + g) - (g + f1)
        assert float(jnp.max(jnp.abs(d(X)))) < 100 * EPS
