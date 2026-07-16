"""Port of MATLAB Chebfun tests/trigtech/test_minus.m (Opus 4.8).

Subtraction of trigtechs / scalars.  For real functions, f-g == -(g-f) and
matches direct evaluation; unhappy operands poison the result.  Array-valued
subtraction (zeros, array-array) now works.  Remaining gaps: empty-argument
arithmetic, complex-scalar subtraction (is_real not cleared, dropping the
imaginary part), 1-column implicit expansion, and scalar-to-row expansion.

Provenance
----------
MATLAB source : tests/trigtech/test_minus.m
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


def _neg_iseq(h1, h2):
    n = max(h1.n, h2.n)
    return _ninf(h1.prolong(n).coeffs + h2.prolong(n).coeffs) == 0.0


# MATLAB's arbitrary constant (seedRNG(6178) draw).
ALPHA = -0.194758928283640 + 0.075474485412665j


class TestTrigtechMinus:
    def test_zeros_minus_zeros(self):
        f = _tt(lambda x: jnp.zeros_like(x))
        h1, h2 = f - f, f - f
        assert _neg_iseq(h1, h2)
        assert _ninf(h1(X)) <= 1e3 * max(h1.vscale * EPS, 0.0)

    def test_sub_function_expcos_and_sin100(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.sin(100 * jnp.pi * x))
        h1, h2 = f - g, g - f
        assert _neg_iseq(h1, h2)
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) - jnp.sin(100 * jnp.pi * X)
        assert _ninf(h1(X) - exact) <= 1e3 * h1.vscale * EPS

    def test_sub_function_expcos_and_sincos(self):
        f = _tt(lambda x: jnp.exp(jnp.cos(jnp.pi * x)) - 1)
        g = _tt(lambda x: jnp.sin(jnp.cos(10 * jnp.pi * x)))
        h1, h2 = f - g, g - f
        assert _neg_iseq(h1, h2)
        exact = (jnp.exp(jnp.cos(jnp.pi * X)) - 1) - jnp.sin(jnp.cos(10 * jnp.pi * X))
        assert _ninf(h1(X) - exact) <= 1e3 * h1.vscale * EPS

    def test_direct_construction_matches(self):
        f = _tt(lambda x: jnp.sin(jnp.pi * jnp.cos(3 * jnp.pi * x)))
        g = _tt(lambda x: jnp.cos(jnp.pi * jnp.sin(10 * jnp.pi * x)) - 1)
        h1 = f - g
        h2 = _tt(lambda x: jnp.sin(jnp.pi * jnp.cos(3 * jnp.pi * x)) - (jnp.cos(jnp.pi * jnp.sin(10 * jnp.pi * x)) - 1))
        n = max(h1.n, h2.n)
        assert _ninf(h1.prolong(n).coeffs - h2.prolong(n).coeffs) < 10 * EPS

    def test_unhappy_minus_happy(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        g = _tt_unhappy(lambda x: jnp.cos(x))
        h = f - g
        assert (not g.ishappy) and (not h.ishappy)

    def test_happy_minus_unhappy(self):
        f = _tt(lambda x: jnp.cos(jnp.pi * x))
        g = _tt_unhappy(lambda x: jnp.cos(x))
        h = g - f
        assert (not g.ishappy) and (not h.ishappy)

    @pytest.mark.xfail(reason="chebfunjax lacks empty-argument arithmetic (raises IndexError)")
    def test_empty_arguments(self):
        raise AssertionError("empty trigtech arithmetic not implemented")

    # FIXED (Fable 5): __sub__ with a complex scalar now clears
    # is_real, so pass(2:3) ports at MATLAB's tolerances.
    def test_sub_complex_scalar(self):
        # pass(2:3): exp(sin(pi x)) - alpha; alpha - f == -(f - alpha).
        fop = lambda x: jnp.exp(jnp.sin(jnp.pi * x))
        f = _tt(fop)
        g1, g2 = f - ALPHA, ALPHA - f
        assert _neg_iseq(g1, g2)
        assert _ninf(g1(X) - (fop(X) - ALPHA)) <= 1e3 * g1.vscale * EPS

    def test_array_zeros(self):
        # pass(10:11): array-valued zeros - zeros.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.zeros_like(x)] * 3, axis=-1))
        h1, h2 = f - f, f - f
        assert _neg_iseq(h1, h2)
        assert _ninf(h1(X)) <= 1e3 * max(h1.vscale * EPS, 0.0)

    # FIXED (Fable 5): complex scalars clear is_real, so pass(12:13)
    # ports directly.
    def test_array_minus_complex_scalar(self):
        # pass(12:13): [sin(10 pi x), sin(cos(pi x)), exp(cos(pi x))] - alpha.
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(jnp.cos(jnp.pi * x)),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1)
        f = _tt(fop)
        g1, g2 = f - ALPHA, ALPHA - f
        assert _neg_iseq(g1, g2)
        assert _ninf(g1(X) - (fop(X) - ALPHA)) <= 1e3 * g1.vscale * EPS

    def test_array_minus_array(self):
        # pass(14:15): array-valued f - array-valued g (g has a complex column).
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(jnp.cos(jnp.pi * x)), jnp.exp(jnp.cos(jnp.pi * x))],
            axis=-1,
        )
        gop = lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.exp(1j * jnp.pi * x) * jnp.exp(1j * jnp.pi * x), jnp.cos(jnp.pi * x)],
            axis=-1,
        )
        f, g = _tt(fop), _tt(gop)
        h1, h2 = f - g, g - f
        assert _neg_iseq(h1, h2)
        assert _ninf(h1(X) - (fop(X) - gop(X))) <= 1e3 * h1.vscale * EPS

    # FIXED (Fable 5): 1-column operands broadcast across (n, m)
    # techs (MATLAB R2016b+ semantics -- pass(16)'s non-error branch,
    # which only requires the subtraction to SUCCEED since sin(10x) is
    # non-periodic).
    def test_dimension_mismatch(self):
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(jnp.cos(jnp.pi * x)),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1)
        f = _tt(fop)
        g = _tt_unhappy(lambda x: jnp.sin(10 * x))
        h = f - g
        assert h.coeffs.shape[1] == 3

    # FIXED (Fable 5): length-m rows expand a scalar tech, so
    # pass(20:21) port directly.
    def test_array_scalar_row(self):
        # pass(20): exp([sin cos -sin^2]) - [1 2 3].
        fop = lambda x: jnp.exp(jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
             -jnp.sin(jnp.pi * x) ** 2], axis=-1))
        f = _tt(fop)
        g = f - jnp.asarray([1.0, 2.0, 3.0])
        exact = fop(X) - jnp.asarray([1.0, 2.0, 3.0])
        assert _ninf(g(X) - exact) < 10 * g.vscale * EPS

    def test_scalar_expansion(self):
        # pass(21): sin(pi x) - [1 2 3] -> 3 columns.
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        g = f - jnp.asarray([1.0, 2.0, 3.0])
        exact = jnp.sin(jnp.pi * X)[:, None] - jnp.asarray([1.0, 2.0, 3.0])
        assert g.coeffs.shape[1] == 3
        assert _ninf(g(X) - exact) < 10 * g.vscale * EPS
