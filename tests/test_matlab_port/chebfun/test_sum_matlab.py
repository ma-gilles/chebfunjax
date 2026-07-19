"""Port of MATLAB Chebfun tests/chebfun/test_sum.m (Fable 5).

Piecewise complex/real integrals at MATLAB tolerances; the subdomain
sum(f, [a b]) is emulated via restrict (MATLAB semantics); transpose/
array-valued variants skipped.

Provenance
----------
MATLAB source : tests/chebfun/test_sum.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)


def _pw():
    # chebfun({exp(4pi i x), exp, exp}, [-1 0 0.5 1])
    def op(x):
        return jnp.where(x < 0, jnp.exp(4j * np.pi * x), jnp.exp(x))
    return cj.chebfun(op, domain=[-1.0, 0.0, 0.5, 1.0])


class TestChebfunSum:
    def test_empty(self):
        from chebfunjax.chebfun1d.chebfun import chebfun
        assert float(chebfun().sum()) == 0.0

    def test_piecewise_complex_total(self):
        f = _pw()
        assert abs(complex(f.sum()) - (np.e - 1)) < 100 * f.vscale * EPS

    def test_subdomain_full(self):
        f = _pw()
        s = complex(f.restrict(-1.0, 1.0).sum())
        assert abs(s - (np.e - 1)) < 100 * f.vscale * EPS

    def test_subdomain_left_is_zero(self):
        f = _pw()
        s = complex(f.restrict(-1.0, 0.0).sum())
        assert abs(s) < 100 * f.vscale * EPS

    def test_subdomain_right(self):
        f = _pw()
        s = complex(f.restrict(0.0, 1.0).sum())
        assert abs(s - (np.e - 1)) < 100 * f.vscale * EPS
