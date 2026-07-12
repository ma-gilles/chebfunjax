"""Port of MATLAB Chebfun tests/chebfun/test_var.m (Fable 5).

FIXED: Chebfun.var/std and Quasimatrix column var/std added in the
Fable 5 audit (array-valued cases via the Quasimatrix counterpart;
the piecewise cell-of-ops construction maps to a jnp.where-selected
callable on the same breakpoints).

Provenance
----------
MATLAB source : tests/chebfun/test_var.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix

EPS = np.finfo(float).eps


class TestChebfunVar:
    def test_piecewise_complex(self):
        # pass(1): var of {exp(4*pi*1i*x), exp, exp} on [-1 0 .5 1]
        f = cj.chebfun(
            lambda x: jnp.where(
                x < 0, jnp.exp(4 * np.pi * 1j * x),
                jnp.exp(x).astype(jnp.complex128)),
            domain=[-1.0, 0.0, 0.5, 1.0])
        assert abs(float(np.real(np.asarray(f.var())))
                   - np.e / 2) < 100 * EPS

    def test_columnwise(self):
        # pass(3): [sin cos exp] column variances
        Q = Quasimatrix(
            [cj.chebfun(jnp.sin), cj.chebfun(jnp.cos),
             cj.chebfun(jnp.exp)],
            cj.chebfun(jnp.sin).domain)
        exact = np.array([
            (1 - np.sin(2) / 2) / 2,
            (1 + np.sin(1) * np.cos(1)) / 2 - np.sin(1) ** 2,
            (1 - np.exp(-2)) / 2])
        assert np.max(np.abs(np.asarray(Q.var()) - exact)) \
            < 100 * EPS
