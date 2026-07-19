"""Port of MATLAB Chebfun tests/chebfun/test_mldivide.m (Fable 5).

FIXED (Fable 5): chebfun-level mldivide (A\\B least squares) over the
scalar / numeric-matrix / quasimatrix cases.  A row/column quasimatrix is
represented here as a Python list of chebfun columns.

Provenance
----------
MATLAB source : tests/chebfun/test_mldivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, mldivide

EPS = float(np.finfo(np.float64).eps)


def _T():
    dom = Domain((-1.0, -0.5, 0.0, 0.5, 1.0))
    return [Chebfun.from_function(
        lambda t, k=k: jnp.cos(k * jnp.arccos(jnp.clip(t, -1.0, 1.0))),
        dom) for k in range(4)]


def _val0(x):
    if isinstance(x, list):
        return np.array([complex(xx(jnp.asarray(0.0))) for xx in x])
    return complex(x(jnp.asarray(0.0)))


class TestChebfunMldivide:
    def test_all_matlab_assertions(self):
        T = _T()

        # pass(1): 2\T == 0.5*T.
        r = mldivide(2.0, T)
        assert max(float((r[k] - T[k] * 0.5).norm()) for k in range(4)) \
            < 10 * EPS

        # pass(2): (1:4)' \ T.'  ->  chebfun, value -1/15 at 0.
        x = mldivide(np.array([1, 2, 3, 4]), T)
        assert abs(_val0(x) - (-1.0 / 15.0)) < 10 * EPS

        # pass(3): eye(4) \ T.'  ->  [1 0 -1 0]' at 0.
        X = mldivide(np.eye(4), T)
        assert np.linalg.norm(_val0(X).real
                              - np.array([1, 0, -1, 0])) < 10 * EPS

        # pass(4): T.' \ (1:4)'  ->  chebfun, value -2.625 at 0.
        x = mldivide(T, np.array([1, 2, 3, 4]))
        assert abs(_val0(x) - (-2.625)) < 100 * EPS
