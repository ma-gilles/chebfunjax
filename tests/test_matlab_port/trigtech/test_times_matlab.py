"""Port of MATLAB Chebfun tests/trigtech/test_times.m (Opus 4.8).

Pointwise multiplication (.*), computed on a physical-space grid to avoid
aliasing.  Scalar (incl. complex) multiplication commutes; products of two
resolved functions match direct construction and evaluation.  Gaps: the
positivity adjustment, conj, empty-argument arithmetic, and array-valued
cases.

Provenance
----------
MATLAB source : tests/trigtech/test_times.m
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
ALPHA = -0.194758928283640 + 0.075474485412665j


def _tt(f):
    return Trigtech.from_function(f)


def _tt_unhappy(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestTrigtechTimes:
    def test_scalar_mult_commutes(self):
        f = _tt(lambda x: jnp.sin(jnp.cos(jnp.pi * x)))
        g1, g2 = f * ALPHA, ALPHA * f
        assert bool(jnp.all(g1.coeffs == g2.coeffs))
        exact = jnp.sin(jnp.cos(jnp.pi * X)) * ALPHA
        assert _ninf(g1(X) - exact) < 200 * g1.vscale * EPS

    def test_mult_by_constant_function(self):
        f = _tt(lambda x: 3.0 / (4 - jnp.cos(jnp.pi * x)))
        g = _tt(lambda x: ALPHA * jnp.ones_like(x))
        h = f * g
        exact = (3.0 / (4 - jnp.cos(jnp.pi * X))) * ALPHA
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_ones_squared(self):
        f = _tt(lambda x: jnp.ones_like(x))
        h = f * f
        assert _ninf(h(X) - jnp.ones_like(X)) < 1e5 * h.vscale * EPS

    def test_mult_expcos_and_rational(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: 3.0 / (4 - jnp.cos(jnp.pi * x)))
        h = f * g
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) * (3.0 / (4 - jnp.cos(jnp.pi * X)))
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_mult_expcos_and_high_freq(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.cos(1e4 * jnp.pi * x))
        h = f * g
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) * jnp.cos(1e4 * jnp.pi * X)
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_mult_expcos_and_complex_exp(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.exp(1j * 1e2 * jnp.pi * x))
        h = f * g
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) * jnp.exp(1j * 1e2 * jnp.pi * X)
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_complex_self_product(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) + jnp.exp(1j * 2 * jnp.pi * x))
        h = f * f
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) + jnp.exp(1j * 2 * jnp.pi * X)) ** 2
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_positivity_norm(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        h = f * f
        exact = (1 + jnp.cos(jnp.pi * X)) ** 2
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_direct_construction_matches(self):
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        g = _tt(lambda x: 3.0 / (4 - jnp.cos(2 * jnp.pi * x)))
        h1 = f * g
        h2 = _tt(lambda x: (1 + jnp.cos(jnp.pi * x)) * 3.0 / (4 - jnp.cos(2 * jnp.pi * x)))
        h2 = h2.prolong(h1.n)
        assert _ninf(h1.coeffs - h2.coeffs) < 50 * EPS

    def test_unhappy_times_happy(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = _tt_unhappy(lambda x: x)
        h = f * g
        assert (not g.ishappy) and (not h.ishappy)

    def test_happy_times_unhappy(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)))
        g = _tt_unhappy(lambda x: x)
        h = g * f
        assert (not g.ishappy) and (not h.ishappy)

    @pytest.mark.xfail(reason="chebfunjax lacks empty-argument arithmetic (raises IndexError)")
    def test_empty_arguments(self):
        raise AssertionError("empty trigtech arithmetic not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax __mul__ does not perform MATLAB's positivity adjustment; "
        "(1+cos)^2 evaluates to a tiny negative value (~-4e-16) at its minimum"
    )
    def test_positivity_nonnegative(self):
        raise AssertionError("positivity adjustment not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no conj() method")
    def test_conj_product_norm(self):
        raise AssertionError("conj() not implemented")

    @pytest.mark.xfail(reason="chebfunjax trigtech has no conj() method")
    def test_conj_product_positivity(self):
        raise AssertionError("conj() not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_scalar_mult(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_times(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued dimension-mismatch detection")
    def test_dimension_mismatch(self):
        raise AssertionError("array-valued trigtech not implemented")
