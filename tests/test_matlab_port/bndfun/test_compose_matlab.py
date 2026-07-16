"""Port of MATLAB Chebfun tests/bndfun/test_compose.m (Opus 4.8).

chebfunjax does not expose ``compose`` at the Bndfun/Classicfun level, but the
underlying ``Chebtech2.compose`` provides it; a Bndfun composition is exactly
``Bndfun.from_chebtech(f.onefun.compose(op[, g]), f.domain)`` -- which is what
MATLAB @classicfun/compose.m does internally (delegate to the onefun, rewrap).
The scalar-valued cases are ported this way and self-validated against the
analytic exact at the SAME tolerance MATLAB uses.  The array-valued cases now
work too (FIXED, Fable 5, Big-Three array-valued epic): the onefun is a
Chebtech2 that supports (n, m) coefficients, so unary, binary, and
tech-of-tech composition all act column-wise.

Provenance
----------
MATLAB source : tests/bndfun/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
XR = np.linspace(-2.0, 7.0, 1000)
X = jnp.asarray(XR)


def _bf(f, dom=DOM, n=None):
    # xfail cases pass a small fixed n so a non-converging build stays fast.
    return Bndfun.from_function(f, dom, n=n)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestBndfunCompose:
    def test_scalar_unary_sin(self):
        # compose(f, @sin) with f = identity  ->  sin(x)
        f = _bf(lambda x: x)
        g = Bndfun.from_chebtech(f.onefun.compose(jnp.sin), f.domain)
        assert _ninf(np.sin(XR) - np.asarray(g(X))) < 10 * g.vscale * EPS

    def test_array_valued_unary_sin_2col(self):
        # pass(2): compose([x x], @sin).
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) onefun compose.
        f = _bf(lambda x: jnp.stack([x, x], axis=-1))
        g = Bndfun.from_chebtech(f.onefun.compose(jnp.sin), f.domain)
        exact = np.stack([np.sin(XR), np.sin(XR)], axis=-1)
        assert _ninf(np.asarray(g(X)) - exact) < 10 * g.vscale * EPS

    def test_array_valued_unary_sin_x_xsq(self):
        # pass(3): compose([x x^2], @sin).
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(lambda x: jnp.stack([x, x ** 2], axis=-1))
        g = Bndfun.from_chebtech(f.onefun.compose(jnp.sin), f.domain)
        exact = np.stack([np.sin(XR), np.sin(XR ** 2)], axis=-1)
        assert _ninf(np.asarray(g(X)) - exact) < 1e2 * g.vscale * EPS

    def test_array_valued_unary_sin_3col(self):
        # pass(4): compose([x x x^2], @sin).
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _bf(lambda x: jnp.stack([x, x, x ** 2], axis=-1))
        g = Bndfun.from_chebtech(f.onefun.compose(jnp.sin), f.domain)
        exact = np.stack([np.sin(XR), np.sin(XR), np.sin(XR ** 2)], axis=-1)
        assert _ninf(np.asarray(g(X)) - exact) < 1e2 * g.vscale * EPS

    def test_binary_plus(self):
        # compose(f1, @plus, f2) = sin(x) + cos(x)
        f1 = _bf(jnp.sin)
        f2 = _bf(jnp.cos)
        g = Bndfun.from_chebtech(f1.onefun.compose(jnp.add, f2.onefun), DOM)
        exact = np.sin(XR) + np.cos(XR)
        assert _ninf(exact - np.asarray(g(X))) < 10 * g.vscale * EPS

    def test_array_valued_binary_times(self):
        # pass(6): compose([sin cos], @times, [cos exp]).
        # FIXED (Fable 5, Big-Three array-valued epic): binary onefun compose.
        f1 = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1))
        f2 = _bf(lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1))
        g = Bndfun.from_chebtech(f1.onefun.compose(jnp.multiply, f2.onefun), DOM)
        exact = np.stack([np.sin(XR) * np.cos(XR), np.cos(XR) * np.exp(XR)], axis=-1)
        assert _ninf(exact - np.asarray(g(X))) < 1e2 * g.vscale * EPS

    def test_function_composition_scalar(self):
        # compose(f, g) = g(f) with f = x^2 (range [0, 49]) and g = sin on [0, 49]
        f = _bf(lambda x: x ** 2)
        g = _bf(jnp.sin, Domain((0.0, 49.0)))
        h = Bndfun.from_chebtech(f.onefun.compose(g), DOM)
        assert _ninf(np.asarray(h(X)) - np.sin(XR ** 2)) < 1e2 * h.vscale * EPS

    def test_function_composition_g_array(self):
        # pass(8): compose(f, g) = g(f) with f = x^2, g = [sin cos] on [0, 49].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued g.
        f = _bf(lambda x: x ** 2)
        g = _bf(lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1), Domain((0.0, 49.0)))
        h = Bndfun.from_chebtech(f.onefun.compose(g), DOM)
        exact = np.stack([np.sin(XR ** 2), np.cos(XR ** 2)], axis=-1)
        assert _ninf(np.asarray(h(X)) - exact) < 1e2 * h.vscale * EPS

    def test_function_composition_f_array(self):
        # pass(9): compose(f, g) = g(f) with f = [x x^2], g = sin on [-2, 49].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued f.
        f = _bf(lambda x: jnp.stack([x, x ** 2], axis=-1))
        g = _bf(jnp.sin, Domain((-2.0, 49.0)))
        h = Bndfun.from_chebtech(f.onefun.compose(g), DOM)
        exact = np.stack([np.sin(XR), np.sin(XR ** 2)], axis=-1)
        assert _ninf(np.asarray(h(X)) - exact) < 1e2 * h.vscale * EPS
