"""Port of MATLAB Chebfun tests/trigtech/test_prolong.m (Opus 4.8).

prolong zero-pads (to a longer grid) or truncates (to a shorter grid) the
Fourier representation; the values on the new equispaced grid must match
the underlying periodic function to machine precision.

Provenance
----------
MATLAB source : tests/trigtech/test_prolong.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech, trigpts

EPS = float(np.finfo(np.float64).eps)


def _F(x):
    return jnp.exp(jnp.sin(jnp.pi * x))


def _tt(f):
    return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechProlong:
    def test_odd_prolongation(self):
        f = _tt(_F)
        k = 101
        go = f.prolong(k)
        x = trigpts(k)
        assert go.n == k
        assert _ninf(go.values - _F(x)) < 10 * go.vscale * EPS

    def test_even_prolongation(self):
        f = _tt(_F)
        k = 100
        ge = f.prolong(k)
        x = trigpts(k)
        assert ge.n == k
        assert _ninf(ge.values - _F(x)) < 10 * ge.vscale * EPS

    def test_odd_restriction_from_odd(self):
        go = _tt(_F).prolong(101)
        k = 89
        g = go.prolong(k)
        x = trigpts(k)
        assert g.n == k
        assert _ninf(g.values - _F(x)) < 10 * g.vscale * EPS

    def test_even_restriction_from_odd(self):
        go = _tt(_F).prolong(101)
        k = 88
        g = go.prolong(k)
        x = trigpts(k)
        assert g.n == k
        assert _ninf(g.values - _F(x)) < 10 * g.vscale * EPS

    def test_odd_restriction_from_even(self):
        ge = _tt(_F).prolong(100)
        k = 89
        g = ge.prolong(k)
        x = trigpts(k)
        assert g.n == k
        assert _ninf(g.values - _F(x)) < 10 * g.vscale * EPS

    def test_even_restriction_from_even(self):
        ge = _tt(_F).prolong(100)
        k = 88
        g = ge.prolong(k)
        x = trigpts(k)
        assert g.n == k
        assert _ninf(g.values - _F(x)) < 10 * g.vscale * EPS

    def test_restriction_to_length_one(self):
        g = _tt(_F).prolong(1)
        assert g.n == 1

    def test_array_valued_prolong(self):
        # pass(8): prolong([F, -F], 101), values match [F(x), -F(x)].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = Trigtech.from_function(lambda x: jnp.stack([_F(x), -_F(x)], axis=-1))
        k = 101
        g = f.prolong(k)
        x = trigpts(k)
        assert g.n == k
        ref = jnp.stack([_F(x), -_F(x)], axis=-1)
        assert _ninf(g.values - ref) < 100 * g.vscale * EPS

    def test_array_valued_prolong_to_one(self):
        # pass(9): prolong([F, -F], 1) has length 1.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Trigtech.from_function(lambda x: jnp.stack([_F(x), -_F(x)], axis=-1))
        g = f.prolong(1)
        assert g.n == 1

    def test_array_valued_same_length(self):
        # pass(10): prolong to the same length leaves the values unchanged.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Trigtech.from_function(lambda x: jnp.stack([_F(x), -_F(x)], axis=-1))
        g = f.prolong(f.n)
        assert bool(jnp.all(f.values == g.values))

    def test_array_valued_values(self):
        # pass(11): a constant array-valued tech [1 2 3] prolonged to 5 has
        # values repmat([1 2 3], 5, 1).
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Trigtech.from_values(jnp.array([[1.0, 2.0, 3.0]]))
        g = f.prolong(5)
        ref = jnp.tile(jnp.array([1.0, 2.0, 3.0]), (5, 1))
        assert _ninf(g.values - ref) < 10 * g.vscale * EPS
