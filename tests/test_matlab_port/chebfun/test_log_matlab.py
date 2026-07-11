"""Port of MATLAB Chebfun tests/chebfun/test_log.m (Fable 5).

log of a positive piecewise base (per-piece construction); the
log10/log1p/log2/reallog variants are skipped (no counterparts).

Provenance
----------
MATLAB source : tests/chebfun/test_log.m
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


def base_op(x):
    return 2 + jnp.sign(x - 0.1) * jnp.abs(x + 0.2) * jnp.sin(3 * x)


def _pw_base():
    ops = [lambda x: 2 + (x + 0.2) * jnp.sin(3 * x),
           lambda x: 2 - (x + 0.2) * jnp.sin(3 * x),
           lambda x: 2 + (x + 0.2) * jnp.sin(3 * x)]
    brks = [-1.0, -0.2, 0.1, 1.0]
    funs = [_Piece.from_function(op, a, b)
            for op, a, b in zip(ops, brks[:-1], brks[1:])]
    return Chebfun(funs=funs, domain=Domain(tuple(brks)))


class TestChebfunLog:
    def test_log_of_piecewise(self):
        f = _pw_base()
        g = f.log()
        exact = jnp.log(base_op(XR))
        mask = (jnp.abs(XR + 0.2) > 1e-6) & (jnp.abs(XR - 0.1) > 1e-6)
        err = jnp.abs(g(XR) - exact)[mask]
        assert float(jnp.max(err)) < 1e2 * max(g.vscale, 1.0) * EPS

    def test_log_variants(self):
        pytest.skip("chebfunjax Chebfun has no log10/log1p/log2/reallog")
