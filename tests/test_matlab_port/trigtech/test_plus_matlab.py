"""Port of MATLAB Chebfun tests/trigtech/test_plus.m (Opus 4.8).

Addition of trigtechs / scalars.  For two real functions, f+g == g+f and
matches direct evaluation; unhappy operands poison the result.  Array-valued
addition works, complex scalars clear is_real, and 1-column/scalar-row
implicit expansion follows MATLAB R2016b+ (all FIXED, Fable 5).  The only
remaining gap is empty-argument arithmetic.

Provenance
----------
MATLAB source : tests/trigtech/test_plus.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np

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



# MATLAB's arbitrary additive constant (seedRNG(6178) draw).
ALPHA = -0.194758928283640 + 0.075474485412665j


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

    def test_empty_arguments(self):
        f = Trigtech.empty()
        g = _tt(lambda x: x)
        assert (f + f).isempty()
        assert (f + g).isempty()
        assert (g + f).isempty()

    # FIXED (Fable 5): __add__ with a complex scalar now clears
    # is_real, so pass(2:5) port at MATLAB's tolerances.
    def test_add_complex_scalar_odd(self):
        # pass(2:3): exp(sin(pi x)) + alpha (odd expansion).
        fop = lambda x: jnp.exp(jnp.sin(jnp.pi * x))
        f = _tt(fop)
        g1, g2 = f + ALPHA, ALPHA + f
        assert _iseq(g1, g2)
        assert _ninf(g1(X) - (fop(X) + ALPHA)) <= 1e3 * g1.vscale * EPS

    def test_add_complex_scalar_even(self):
        # pass(4:5): even-length coefficient construction + alpha.
        # MATLAB forces even coeffs via make({[],[1 1 0 1]'}).
        f = Trigtech.from_coeffs(
            jnp.asarray([1.0, 1.0, 0.0, 1.0], dtype=jnp.complex128))
        fx = f(X)
        g1, g2 = f + ALPHA, ALPHA + f
        assert _iseq(g1, g2)
        assert _ninf(g1(X) - (fx + ALPHA)) <= 1e3 * g1.vscale * EPS

    def test_array_zeros(self):
        # pass(12:13): array-valued zeros + zeros.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.zeros_like(x)] * 3, axis=-1))
        h1, h2 = f + f, f + f
        assert _iseq(h1, h2)
        assert _ninf(h1(X)) <= 1e3 * max(h1.vscale * EPS, 0.0)

    # FIXED (Fable 5): complex scalars clear is_real, so pass(14:15)
    # ports directly.
    def test_array_plus_complex_scalar(self):
        # pass(14:15): [sin(10 pi x), sin(cos(pi x)), exp(cos(pi x))] + alpha.
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(jnp.cos(jnp.pi * x)),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1)
        f = _tt(fop)
        g1, g2 = f + ALPHA, ALPHA + f
        assert _iseq(g1, g2)
        assert _ninf(g1(X) - (fop(X) + ALPHA)) <= 1e3 * g1.vscale * EPS

    def test_array_plus_array(self):
        # pass(16:17): array-valued f + array-valued g (g has a complex column).
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
        h1, h2 = f + g, g + f
        assert _iseq(h1, h2)
        assert _ninf(h1(X) - (fop(X) + gop(X))) <= 1e3 * h1.vscale * EPS

    # FIXED (Fable 5): 1-column operands broadcast across (n, m)
    # techs (MATLAB R2016b+ semantics -- pass(18)'s non-error branch).
    def test_dimension_mismatch(self):
        # pass(18): 3-column f + scalar-valued g broadcasts.  MATLAB's
        # R2016b+ branch only requires the addition to SUCCEED (g =
        # sin(10x) is non-periodic, so no accuracy is expected of its
        # trig representation; MATLAB asserts pass(18) = true with no
        # value check).
        fop = lambda x: jnp.stack(
            [jnp.sin(10 * jnp.pi * x), jnp.sin(jnp.cos(jnp.pi * x)),
             jnp.exp(jnp.cos(jnp.pi * x))], axis=-1)
        gop = lambda x: jnp.sin(10 * x)
        f, g = _tt(fop), _tt_unhappy(gop)
        h = f + g
        assert h.coeffs.shape[1] == 3

    # FIXED (Fable 5): a length-m row expands a scalar tech to m
    # columns, so pass(22:23) port directly.
    def test_array_scalar_row(self):
        # pass(22): array-valued tech + [1 2 3].
        fop = lambda x: jnp.exp(jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x),
             -jnp.sin(jnp.pi * x) ** 2], axis=-1))
        f = _tt(fop)
        g = f + jnp.asarray([1.0, 2.0, 3.0])
        exact = fop(X) + jnp.asarray([1.0, 2.0, 3.0])
        assert _ninf(g(X) - exact) < 10 * g.vscale * EPS

    def test_scalar_expansion(self):
        # pass(23): sin(pi x) + [1 2 3] -> 3 columns.
        f = _tt(lambda x: jnp.sin(jnp.pi * x))
        g = f + jnp.asarray([1.0, 2.0, 3.0])
        exact = jnp.sin(jnp.pi * X)[:, None] + jnp.asarray([1.0, 2.0, 3.0])
        assert g.coeffs.shape[1] == 3
        assert _ninf(g(X) - exact) < 10 * g.vscale * EPS
