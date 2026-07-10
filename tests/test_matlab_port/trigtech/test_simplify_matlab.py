"""Port of MATLAB Chebfun tests/trigtech/test_simplify.m (Opus 4.8).

simplify removes negligible trailing Fourier coefficients.  Empty and
unhappy trigtechs are left untouched; the simplified length is invariant
under vertical scaling; and a resolved function simplifies to a strictly
shorter representation that still matches to the requested tolerance.

Provenance
----------
MATLAB source : tests/trigtech/test_simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

EPS = float(np.finfo(np.float64).eps)
X = jnp.asarray(np.linspace(-1.0, 1.0, 100, endpoint=False))
SIMPTOL = 1e-6


def _tt(f):
    return Trigtech.from_function(f)


def _tt_unhappy(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _iszero(f):
    return _ninf(f.coeffs) == 0.0


class TestTrigtechSimplify:
    def test_empty_left_alone(self):
        f = Trigtech(coeffs=jnp.array([], dtype=jnp.complex128))
        g = f.simplify()
        assert g.n == f.n == 0

    def test_unhappy_left_alone(self):
        f = _tt_unhappy(lambda x: jnp.sin(x))  # not periodic -> unhappy
        g = f.simplify()
        assert not f.ishappy
        assert _ninf(g.coeffs - f.coeffs) == 0.0 and g.n == f.n

    def test_scalar_leading_coeff_nonzero(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)) + jnp.exp(jnp.cos(3 * jnp.pi * x)))
        g = f.simplify(SIMPTOL)
        assert abs(complex(g.coeffs[-1])) != 0.0

    def test_scalar_shorter(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)) + jnp.exp(jnp.cos(3 * jnp.pi * x)))
        g = f.simplify(SIMPTOL)
        assert g.n < f.n

    def test_scalar_accuracy(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)) + jnp.exp(jnp.cos(3 * jnp.pi * x)))
        g = f.simplify(SIMPTOL)
        assert _ninf(f(X) - g(X)) < 1e1 * SIMPTOL * f.vscale

    def test_scale_invariance_small(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)) + jnp.exp(jnp.cos(3 * jnp.pi * x)))
        g = f.simplify(SIMPTOL)
        g1 = (1e-8 * f).simplify(SIMPTOL)
        assert g1.n == g.n

    def test_scale_invariance_large(self):
        f = _tt(lambda x: jnp.exp(jnp.sin(2 * jnp.pi * x)) + jnp.exp(jnp.cos(3 * jnp.pi * x)))
        g = f.simplify(SIMPTOL)
        g2 = (1e8 * f).simplify(SIMPTOL)
        assert g2.n == g.n

    def test_contrived_length_one(self):
        f = _tt(lambda x: jnp.sin(100 * jnp.pi * (x + 0.1)))
        g = f.simplify(1e20)
        assert g.n == 1

    def test_zero_simplifies_to_one(self):
        f = Trigtech.from_function(lambda x: 0 * x, n=8)
        g = f.simplify()
        assert _iszero(g) and g.n == 1

    def test_even_ones_simplify(self):
        f = Trigtech.from_values(jnp.ones(8))
        g = f.simplify()
        assert _ninf(g.values - 1.0) < 10 * EPS

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_leading_coeff(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_shorter(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_accuracy(self):
        raise AssertionError("array-valued trigtech not implemented")
