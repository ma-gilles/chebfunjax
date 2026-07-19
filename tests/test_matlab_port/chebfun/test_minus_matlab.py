"""Port of MATLAB Chebfun tests/chebfun/test_minus.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_minus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

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


class TestChebfunMinus:
    def test_empty_cases(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        f = chebfun(lambda x: jnp.sin(x))
        g = chebfun()
        assert (f - []).isempty()
        assert (f - g).isempty()

    def test_subtract_scalar(self):
        f1 = _kinked()
        g = f1 - ALPHA
        exact = jnp.sin(X) * jnp.abs(X - 0.1) - ALPHA
        assert float(jnp.max(jnp.abs(g(X) - exact))) \
            < 100 * f1.vscale * EPS

    def test_rsub(self):
        f1 = _kinked()
        g = ALPHA - f1
        exact = ALPHA - jnp.sin(X) * jnp.abs(X - 0.1)
        assert float(jnp.max(jnp.abs(g(X) - exact))) \
            < 100 * f1.vscale * EPS

    def test_f_minus_f_is_zero(self):
        f1 = _kinked()
        d = f1 - f1
        assert float(jnp.max(jnp.abs(d(X)))) < 100 * EPS
