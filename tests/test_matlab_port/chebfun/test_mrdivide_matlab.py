"""Port of MATLAB Chebfun tests/chebfun/test_mrdivide.m (Fable 5).

FIXED (Fable 5): chebfun-level mrdivide (A/B least squares), the
transpose-dual of mldivide.  A quasimatrix is a Python list of chebfun
columns.

Provenance
----------
MATLAB source : tests/chebfun/test_mrdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import legendre as _leg

from chebfunjax.chebfun1d.chebfun import Chebfun, Domain, mrdivide

EPS = float(np.finfo(np.float64).eps)


def _T():
    dom = Domain((-1.0, -0.5, 0.0, 0.5, 1.0))
    return [Chebfun.from_function(
        lambda t, k=k: jnp.cos(k * jnp.arccos(jnp.clip(t, -1.0, 1.0))),
        dom) for k in range(4)]


def _L():
    dom = Domain((-1.0, 0.0, 1.0))
    out = []
    for k in range(4):
        c = np.zeros(k + 1)
        c[k] = 1.0
        out.append(Chebfun.from_function(
            lambda t, c=c: jnp.asarray(_leg.legval(np.asarray(t), c)), dom))
    return out


def _val0(x):
    if isinstance(x, list):
        return np.array([complex(xx(jnp.asarray(0.0))) for xx in x])
    return complex(x(jnp.asarray(0.0)))


class TestChebfunMrdivide:
    def test_all_matlab_assertions(self):
        T = _T()

        # pass(1): T/2 == 0.5*T.
        r = mrdivide(T, 2.0)
        assert max(float((r[k] - T[k] * 0.5).norm()) for k in range(4)) \
            < 10 * EPS

        # pass(2): (1:4) / T  ->  chebfun, value -2.625 at 0.
        x = mrdivide(np.array([1, 2, 3, 4]), T)
        assert abs(_val0(x) - (-2.625)) < 100 * EPS

        # pass(3): eye(4) / L  ->  [.5 0 -1.25 0]' at 0.
        X = mrdivide(np.eye(4), _L())
        assert np.linalg.norm(_val0(X).real
                              - np.array([0.5, 0, -1.25, 0])) < 1e2 * EPS

        # pass(4): T / (1:4)  ->  chebfun, value -1/15 at 0.
        x = mrdivide(T, np.array([1, 2, 3, 4]))
        assert abs(_val0(x) - (-1.0 / 15.0)) < 10 * EPS
