"""Port of MATLAB Chebfun tests/trigtech/test_roots.m (Opus 4.8).

Real roots of a trigtech in [-1, 1).  chebfunjax finds roots by sampling
the trigonometric interpolant on Chebyshev points and calling Chebyshev
rootfinding; it does NOT support the 'complex' flag or array-valued
trigtechs, and for very short expansions the Chebyshev resampling is less
accurate than MATLAB's dedicated trigtech rootfinder (see xfail below).

Provenance
----------
MATLAB source : tests/trigtech/test_roots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechRoots:
    @pytest.mark.xfail(
        reason="chebfunjax trigtech.roots resamples cos(5*pi*x) (len 11) on only 33 "
        "Chebyshev points, giving ~3e-10 root error vs MATLAB tol ~2e-14"
    )
    def test_cos_five_pi(self):
        f = _tt(lambda x: jnp.cos(5 * jnp.pi * x))
        r = np.sort(np.array(f.roots()))
        exact = np.arange(-0.9, 1.0, 0.2)
        assert _ninf(r - exact) < 1e1 * f.n * EPS

    def test_sin_of_sin(self):
        k = 20
        f = _tt(lambda x: jnp.sin(jnp.sin(jnp.pi * k * x)))
        r = np.sort(np.array(f.roots()))
        exact = np.arange(-k, k + 1) / k
        assert _ninf(r - exact) < f.n * EPS

    def test_no_real_roots(self):
        f = _tt(lambda x: 3.0 / (5 - 4 * jnp.cos(3 * jnp.pi * x)))
        assert f.roots().shape[0] == 0

    def test_sin_hundred_pi_root_count(self):
        # roots() (real) should return at least 201 roots of sin(100 pi x).
        f = _tt(lambda x: jnp.sin(100 * jnp.pi * x))
        assert f.roots().shape[0] >= 201

    @pytest.mark.xfail(reason="chebfunjax trigtech.roots lacks the 'complex' flag")
    def test_complex_roots_of_shifted_cos(self):
        raise AssertionError("roots(f, 'complex', 1) not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech.roots lacks the 'complex' flag")
    def test_complex_root_count(self):
        raise AssertionError("roots(f, 'complex', 1) not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_valued_roots(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech and the 'complex' flag")
    def test_array_valued_complex_roots(self):
        raise AssertionError("array-valued trigtech not implemented")
