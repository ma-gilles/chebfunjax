"""Port of MATLAB Chebfun tests/chebfun/test_pinv.m (Fable 5).

FIXED: Quasimatrix.pinv added in the Fable 5 audit, with
complex-capable Householder QR/SVD.  The MATLAB test uses an
array-valued chebfun on a split domain [-1 0 1]; the chebfunjax
counterpart is a Quasimatrix on the single interval [-1, 1] (the
quasimatrix layer requires a single-interval domain), which carries
the same assertions.

Provenance
----------
MATLAB source : tests/chebfun/test_pinv.m
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


class TestChebfunPinv:
    def test_least_squares_in_range(self):
        # pass(1)-(2): pinv gives the exact solve when f is in range
        A = _build(False)
        fx = A.cols[0] * 2.0 + A.cols[2] * (-0.7)
        x = np.asarray(A.pinv(fx))
        rec = A @ x
        assert float(jnp.max(jnp.abs(rec.cols[0](XS) - fx(XS)))) \
            < 100 * EPS

    def test_complex_least_squares(self):
        # pass(3)-(4): complex quasimatrix
        A = _build(True)
        fx = A.cols[0] * 2.0 + A.cols[2] * (1 + 0.5j)
        x = np.asarray(A.pinv(fx))
        rec = A @ x
        assert float(jnp.max(jnp.abs(rec.cols[0](XS) - fx(XS)))) \
            < 100 * EPS

    def test_out_of_range_projection(self):
        # least squares onto {1, x}: argmin ||a + b x - (1+2x+x^2)||
        one = cj.chebfun(lambda t: t ** 0)
        x = cj.chebfun(lambda t: t)
        B = Quasimatrix([one, x], one.domain)
        fx = cj.chebfun(lambda t: 1 + 2 * t + t ** 2)
        c = np.asarray(B.pinv(fx))
        assert np.max(np.abs(c - np.array([4.0 / 3.0, 2.0]))) \
            < 100 * EPS
