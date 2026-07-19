"""Port of MATLAB Chebfun tests/singfun/test_chebcoeffs.m (Opus 4.8).

MATLAB ``chebcoeffs(f, N)`` returns the first N Chebyshev coefficients of the
*singular* function f (a projection that accounts for the endpoint weights).
chebfunjax Singfun exposes ``f.coeffs`` (the smooth-part coefficients) but has
no ``chebcoeffs`` projection method, so every assertion is xfailed.

Provenance
----------
MATLAB source : tests/singfun/test_chebcoeffs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

_REASON = "chebfunjax Singfun has no chebcoeffs() projection method"


def _sf(f, exps):
    return Singfun.from_function(f, exps)


class TestSingfunChebcoeffs:
    def test_length_first_kind(self):
        f = _sf(lambda x: jnp.sqrt(1 - x), (0.0, 0.5))
        c = f.chebcoeffs(10)
        assert len(c) == 10

    def test_values_first_kind(self):
        f = _sf(lambda x: jnp.sqrt(1 - x), (0.0, 0.5))
        c = f.chebcoeffs(10)
        exact = np.array([np.sqrt(2), -2 * np.sqrt(2) / 35]) * (2 / np.pi)
        got = np.asarray(c)[[0, 3]]
        assert np.linalg.norm(got - exact) < 10 * EPS

    def test_length_second_kind(self):
        f = _sf(lambda x: (1 + x ** 2 + x ** 3) * jnp.sqrt(1 - x ** 2), (0.5, 0.5))
        c = f.chebcoeffs(10, kind=2)
        assert len(c) == 10

    def test_values_second_kind(self):
        f = _sf(lambda x: (1 + x ** 2 + x ** 3) * jnp.sqrt(1 - x ** 2), (0.5, 0.5))
        c = f.chebcoeffs(10, kind=2)
        exact = np.array([8 / 5, 16 / 315]) * (2 / np.pi)
        got = np.asarray(c)[[0, 3]]
        assert np.linalg.norm(got - exact) < 10 * EPS
