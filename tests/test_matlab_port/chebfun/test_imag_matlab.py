"""Port of MATLAB Chebfun tests/chebfun/test_imag.m (Fable 5).

The kinked complex function is built per-piece (MATLAB splitting
equivalent), as in the exp/plus ports.

Provenance
----------
MATLAB source : tests/chebfun/test_imag.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain

EPS = float(np.finfo(np.float64).eps)
RNG = np.random.default_rng(7681)
XR = jnp.asarray(2 * RNG.uniform(size=1000) - 1)


def _kinked_complex():
    funs = [_Piece.from_function(
                lambda x: -jnp.exp(1j * x) * (x - 0.1), -1.0, 0.1),
            _Piece.from_function(
                lambda x: jnp.exp(1j * x) * (x - 0.1), 0.1, 1.0)]
    return Chebfun(funs=funs, domain=Domain((-1.0, 0.1, 1.0)))


class TestChebfunImag:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_imag_of_kinked_complex(self):
        f = _kinked_complex()
        imf = f.imag()
        exact = jnp.sin(XR) * jnp.abs(XR - 0.1)
        err = jnp.abs(imf(XR) - exact)
        assert float(jnp.max(err)) < 100 * EPS

    def test_real_of_kinked_complex(self):
        f = _kinked_complex()
        ref = f.real()
        exact = jnp.cos(XR) * jnp.abs(XR - 0.1)
        err = jnp.abs(ref(XR) - exact)
        assert float(jnp.max(err)) < 100 * EPS
