"""Port of MATLAB Chebfun tests/trigtech/test_plus.m (Opus 4.8).

Addition of trigtechs / scalars.  For two real functions, f+g == g+f and
matches direct evaluation; unhappy operands poison the result.  chebfunjax
gaps flagged below: empty-argument arithmetic, complex-scalar addition
(is_real is not cleared), and all array-valued cases.

Provenance
----------
MATLAB source : tests/trigtech/test_plus.m
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


def _tt(f):
    return Trigtech.from_function(f)


def _tt_unhappy(f):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Trigtech.from_function(f)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


def _iseq(f, g):
    return f.n == g.n and bool(jnp.all(f.coeffs == g.coeffs))


class TestTrigtechPlus:
    def test_zeros_plus_zeros(self):
        f = _tt(lambda x: jnp.zeros_like(x))
        h1, h2 = f + f, f + f
        assert _iseq(h1, h2)
        assert _ninf(h1(X)) <= 1e3 * max(h1.vscale * EPS, 0.0)

    def test_add_function_expcos_and_sin100(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.sin(100 * jnp.pi * x))
        h1, h2 = f + g, g + f
        assert _iseq(h1, h2)
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) + jnp.sin(100 * jnp.pi * X)
        assert _ninf(h1(X) - exact) <= 1e3 * h1.vscale * EPS

    def test_add_function_expcos_and_sincos(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.sin(jnp.cos(10 * jnp.pi * x)))
        h1, h2 = f + g, g + f
        assert _iseq(h1, h2)
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) + jnp.sin(jnp.cos(10 * jnp.pi * X))
        assert _ninf(h1(X) - exact) <= 1e3 * h1.vscale * EPS

    def test_direct_construction_matches(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * jnp.cos(3 * jnp.pi * x)))
        g = _tt(lambda x: jnp.cos(jnp.pi * jnp.sin(10 * jnp.pi * x)) - 1)
        h1 = f + g
        h2 = _tt(lambda x: jnp.sin(jnp.pi * jnp.cos(3 * jnp.pi * x)) + (jnp.cos(jnp.pi * jnp.sin(10 * jnp.pi * x)) - 1))
        n = max(h1.n, h2.n)
        assert _ninf(h1.prolong(n).coeffs - h2.prolong(n).coeffs) < 10 * EPS

    def test_unhappy_plus_happy(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        g = _tt_unhappy(lambda x: jnp.cos(x))
        h = f + g
        assert (not g.ishappy) and (not h.ishappy)

    def test_happy_plus_unhappy(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        g = _tt_unhappy(lambda x: jnp.cos(x))
        h = g + f
        assert (not g.ishappy) and (not h.ishappy)

    @pytest.mark.xfail(reason="chebfunjax lacks empty-argument arithmetic (raises IndexError)")
    def test_empty_arguments(self):
        raise AssertionError("empty trigtech arithmetic not implemented")

    @pytest.mark.xfail(
        reason="chebfunjax __add__ with a complex scalar does not clear is_real, so the "
        "imaginary part is dropped on evaluation"
    )
    def test_add_complex_scalar_odd(self):
        raise AssertionError("complex-scalar addition drops imaginary part")

    @pytest.mark.xfail(
        reason="chebfunjax __add__ with a complex scalar does not clear is_real, so the "
        "imaginary part is dropped on evaluation"
    )
    def test_add_complex_scalar_even(self):
        raise AssertionError("complex-scalar addition drops imaginary part")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_zeros(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_plus_complex_scalar(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued (multi-column) trigtech")
    def test_array_plus_array(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued dimension-mismatch detection")
    def test_dimension_mismatch(self):
        raise AssertionError("array-valued trigtech not implemented")

    @pytest.mark.xfail(reason="chebfunjax lacks array-valued trigtech and scalar expansion")
    def test_scalar_expansion(self):
        raise AssertionError("array-valued trigtech not implemented")
