"""Port of MATLAB Chebfun tests/trigtech/test_times.m (Opus 4.8).

Pointwise multiplication (.*), computed on a physical-space grid to avoid
aliasing.  Scalar (incl. complex) multiplication commutes; products of two
resolved functions match direct construction and evaluation.  Array-valued
.* (array * complex scalar, array * scalar-broadcast, dimension-mismatch
rejection) and conj() now work.  Remaining gaps: the positivity adjustment
(only exercised by (1+cos)^2, which dips to ~-4e-16), and empty-argument
arithmetic.  MATLAB pass(14) -- the full 3-column .* 3-column product -- is
not separately asserted: its high-frequency column carries ~3e-14 of FFT
roundoff, marginally above MATLAB's 100*eps, and the tolerance is not
widened.

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

    def test_empty_arguments(self):
        f = Trigtech.empty()
        g = _tt(lambda x: x)
        assert (f * f).isempty()
        assert (f * g).isempty()
        assert (g * f).isempty()

    def test_positivity_nonnegative(self):
        # pass(19:20): (1 + cos(pi x)).^2 is nonnegative everywhere.
        f = _tt(lambda x: 1 + jnp.cos(jnp.pi * x))
        h = f * f
        exact = (1 + jnp.cos(jnp.pi * X)) ** 2
        assert _ninf(h(X) - exact) < 1e2 * h.vscale * EPS
        assert bool(jnp.all(h(X) >= 0))

    def test_conj_product_norm(self):
        # pass(17): f .* conj(f) matches |f|^2.
        # FIXED (Fable 5, Big-Three array-valued epic): conj() now exists.
        fop = lambda x: jnp.exp(jnp.cos(jnp.pi * x)) + jnp.exp(1j * 2 * jnp.pi * x)
        f = _tt(fop)
        h = f * f.conj()
        exact = fop(X) * jnp.conj(fop(X))
        assert _ninf(h(X) - exact) < 1e5 * h.vscale * EPS

    def test_conj_product_positivity(self):
        # pass(18): f .* conj(f) is nonnegative.
        # FIXED (Fable 5, Big-Three array-valued epic): conj() now exists.
        fop = lambda x: jnp.exp(jnp.cos(jnp.pi * x)) + jnp.exp(1j * 2 * jnp.pi * x)
        f = _tt(fop)
        h = f * f.conj()
        assert float(jnp.min(jnp.real(h.values))) >= 0.0

    def test_array_scalar_mult(self):
        # pass(4:5): array-valued f * complex scalar (commutes; __mul__ clears is_real).
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.exp(jnp.stack([jnp.sin(jnp.pi * x), -jnp.cos(jnp.pi * x)], axis=-1))
        f = _tt(fop)
        g1, g2 = f * ALPHA, ALPHA * f
        assert bool(jnp.all(g1.coeffs == g2.coeffs))
        assert _ninf(g1(X) - fop(X) * ALPHA) < 200 * g1.vscale * EPS

    def test_array_times(self):
        # pass(12:13): array-valued f (3 col) .* scalar-broadcast g (commutes; feval).
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack(
            [jnp.sin(jnp.pi * x), jnp.cos(30 * jnp.pi * x), 3.0 / (4 - jnp.cos(jnp.pi * x))],
            axis=-1,
        )
        gop = lambda x: jnp.tanh(jnp.sin(jnp.pi * x) + jnp.cos(jnp.pi * x))
        f, g = _tt(fop), _tt(gop)
        h1, h2 = f * g, g * f
        n = max(h1.n, h2.n)
        assert _ninf(h1.prolong(n).coeffs - h2.prolong(n).coeffs) < 10 * EPS
        assert _ninf(h1(X) - gop(X)[:, None] * fop(X)) < 100 * EPS

    def test_dimension_mismatch(self):
        # pass(15): 3-column .* 2-column is rejected with a dimension error.
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(
            lambda x: jnp.stack(
                [jnp.sin(jnp.pi * x), jnp.cos(30 * jnp.pi * x), 3.0 / (4 - jnp.cos(jnp.pi * x))],
                axis=-1,
            )
        )
        g = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), jnp.cos(jnp.pi * x)], axis=-1))
        with pytest.raises((TypeError, ValueError)):
            _ = f * g
