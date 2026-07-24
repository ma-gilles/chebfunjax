"""Port of MATLAB Chebfun tests/chebfun/test_constructor_turbo.m (Opus 4.8).

The ``'turbo'`` flag recomputes the Chebyshev coefficients to high accuracy
via the turbo (Bernstein-ellipse contour-integral) constructor.  MATLAB
checks basic construction syntax only: a turbo construction has twice the
plain length (or the exact requested length when ``n`` is given).  The tech
already implements turbo; this file pins the flag wired through the
``chebfun`` factory for the adaptive, array-valued, fixed-``n``, broken-domain,
splitting, and exponent construction paths.

Provenance
----------
MATLAB source : tests/chebfun/test_constructor_turbo.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp

import chebfunjax as cj

_ARR = lambda x: jnp.stack([jnp.exp(x), 1.0 / (x + 5)], axis=-1)  # noqa: E731


class TestChebfunConstructorTurbo:
    def test_adaptive_doubles_length(self):
        # pass(1): turbo doubles the number of coefficients of @exp.
        f_plain = cj.chebfun(jnp.exp)
        f_turbo = cj.chebfun(jnp.exp, turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    def test_array_valued_doubles_length(self):
        # pass(2): same for an array-valued input [exp(x), 1/(x+5)].
        f_plain = cj.chebfun(_ARR)
        f_turbo = cj.chebfun(_ARR, turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    def test_fixed_length(self):
        # pass(3): turbo with a requested length keeps exactly that many.
        f = cj.chebfun(jnp.exp, n=75, turbo=True)
        assert len(f) == 75

    def test_fixed_length_array_valued(self):
        # pass(4): fixed length for an array-valued input.
        f = cj.chebfun(_ARR, n=75, turbo=True)
        assert len(f) == 75

    def test_breakpoints_double_length(self):
        # pass(5): construction with breakpoints -- each piece doubles, so the
        # total (summed) length doubles.
        f_plain = cj.chebfun(jnp.exp, domain=(-1, 0, 1))
        f_turbo = cj.chebfun(jnp.exp, domain=(-1, 0, 1), turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    def test_splitting_double_length(self):
        # pass(6): turbo alongside splitting-on for exp(x)*sign(x).
        import warnings
        op = lambda x: jnp.exp(x) * jnp.sign(x)  # noqa: E731
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_plain = cj.chebfun(op, splitting=True)
            f_turbo = cj.chebfun(op, splitting=True, turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    def test_exponents_double_length(self):
        # pass(7): turbo alongside an 'exps' (SingFun) construction; the smooth
        # part is what doubles.
        op = lambda x: jnp.sin(x) * jnp.sqrt(1 + x)  # noqa: E731
        f_plain = cj.chebfun(op, exps=(0.5, 0.0))
        f_turbo = cj.chebfun(op, exps=(0.5, 0.0), turbo=True)
        assert len(f_turbo) == 2 * len(f_plain)

    def test_turbo_accuracy(self):
        # A turbo construction must still approximate the function (the extra
        # coefficients are computed to high accuracy, not padding).
        f = cj.chebfun(jnp.exp, turbo=True)
        xs = jnp.linspace(-1, 1, 50)
        assert float(jnp.max(jnp.abs(f(xs) - jnp.exp(xs)))) < 1e-14
