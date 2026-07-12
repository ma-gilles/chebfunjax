"""Port of MATLAB Chebfun tests/chebfun/test_null.m (Fable 5).

FIXED: Quasimatrix.null added in the Fable 5 audit, with
complex-capable Householder QR/SVD.  The MATLAB test uses an
array-valued chebfun on a split domain [-1 0 1]; the chebfunjax
counterpart is a Quasimatrix on the single interval [-1, 1] (the
quasimatrix layer requires a single-interval domain), which carries
the same assertions.

Provenance
----------
MATLAB source : tests/chebfun/test_null.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = np.finfo(float).eps
XS = jnp.asarray(np.linspace(-0.95, 0.95, 33))


def _build(cplx):
    s = cj.chebfun(jnp.sin)
    c = cj.chebfun(jnp.cos)
    third = cj.chebfun(lambda x: jnp.exp(1j * x)) if cplx \
        else cj.chebfun(jnp.exp)
    return Quasimatrix([s, c, third], s.domain)


class TestChebfunNull:
    def test_full_rank_empty_null(self):
        # pass(1): [sin cos exp] is full rank
        f = _build(False)
        Z = np.asarray(f.null())
        assert Z.shape[1] == 0

    def test_complex_rank_deficient(self):
        # pass(2)-(4): [sin cos exp(1i*x)] has a 1-dim null space
        f = _build(True)
        Z = np.asarray(f.null())
        assert Z.shape == (3, 1)
        v = Z[:, 0]
        assert abs(np.vdot(v, v) - 1) < 10 * EPS
        resid = (f.cols[0] * complex(v[0])
                 + f.cols[1] * complex(v[1])
                 + f.cols[2] * complex(v[2]))
        assert float(jnp.max(jnp.abs(resid(XS)))) < 100 * EPS
