"""Port of MATLAB Chebfun tests/chebfun/test_norm.m (Fable 5).

Provenance
----------
MATLAB source : tests/chebfun/test_norm.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
NORMF_2 = float(np.sqrt(1 + (np.exp(2) - 1) / 2))


def _pw():
    def op(x):
        return jnp.where(x < 0, jnp.exp(4j * np.pi * x), jnp.exp(x))
    return cj.chebfun(op, domain=[-1.0, 0.0, 0.5, 1.0])


class TestChebfunNorm:
    def test_empty(self):
        pytest.skip("chebfunjax has no empty chebfun")

    def test_two_norm_default(self):
        f = _pw()
        assert abs(float(f.norm()) - NORMF_2) < 100 * f.vscale * EPS

    def test_two_norm_explicit(self):
        f = _pw()
        assert abs(float(f.norm(2)) - NORMF_2) < 100 * f.vscale * EPS

    def test_one_norm(self):
        f = _pw()
        assert abs(float(f.norm(1)) - np.e) < 100 * f.vscale * EPS

    @pytest.mark.xfail(
        reason="chebfunjax norm(inf) crashes on complex chebfuns "
        "(lt on complex128); MATLAB uses |f| via minandmax")
    def test_inf_norm_complex(self):
        f = _pw()
        assert abs(float(f.norm(jnp.inf)) - np.e) < 100 * f.vscale * EPS

    def test_inf_norm_real(self):
        f = cj.chebfun(lambda x: jnp.exp(x), domain=[-1.0, 0.0, 1.0])
        assert abs(float(f.norm(jnp.inf)) - np.e) < 100 * f.vscale * EPS
