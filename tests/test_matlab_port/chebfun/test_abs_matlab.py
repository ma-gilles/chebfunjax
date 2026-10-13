"""Smooth scalar, real, imaginary, complex, array and pointValues clauses
from MATLAB Chebfun tests/chebfun/test_abs.m.

Singular, quasimatrix, unbounded, and trig source clauses are not asserted here.

Provenance
----------
MATLAB source : tests/chebfun/test_abs.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
_FIXTURE = Path(__file__).with_name("fixtures") / "abs_seed6178_native.json"
with _FIXTURE.open(encoding="utf-8") as _stream:
    _NATIVE_RANDOM = json.load(_stream)
X = jnp.asarray(_NATIVE_RANDOM["x"], dtype=jnp.float64)
X2 = jnp.asarray(_NATIVE_RANDOM["x2"], dtype=jnp.float64)
X3 = jnp.asarray(_NATIVE_RANDOM["x3"], dtype=jnp.float64)
BREAKS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _native_reference_f():
    """Four source callbacks used by test_abs.m's fHandle cell array."""
    sine = lambda x: jnp.sin(jnp.pi * x)
    return cj.chebfun([sine, lambda x: -sine(x), sine,
                       lambda x: -sine(x)], domain=BREAKS)


class TestChebfunAbs:
    def test_abs_of_square_piece_count(self):
        f = cj.chebfun(lambda x: x ** 2)
        f1 = f.abs()
        assert len(f1.funs) in (1, 2)

    def test_abs_nonnegative(self):
        f = cj.chebfun(lambda x: jnp.cos(3 * x))
        f1 = f.abs()
        assert bool(jnp.all(f1(X) >= 0))

    def test_abs_values_match(self):
        # Supplemental pointwise control for the source scalar positivity path.
        f = cj.chebfun(lambda x: jnp.cos(3 * x))
        f1 = f.abs()
        err = jnp.abs(f1(X) - jnp.abs(jnp.cos(3 * X)))
        assert float(jnp.max(err)) < 10 * f.vscale * EPS

    def test_point_values(self):
        # Native pass(1,3): construct cos(3x), add 2, restrict to sorted x,
        # then assign the original unsorted x as pointValues.
        ordered = np.sort(np.asarray(X))
        f = cj.chebfun(lambda x: jnp.cos(3 * x))
        f = (f + 2.0).restrict(ordered)
        f = f.set_point_values(X)
        f1 = f.abs()
        assert bool(jnp.all(f1.point_values == jnp.abs(X)))

    def test_native_scalar_abs_predicates(self):
        # Native pass(1,4): positivity at the exact second native RNG draw.
        long_f = cj.chebfun(lambda x: x, domain=(-1.0, 100.0))
        assert bool(jnp.all(long_f.abs()(X2) >= 0))

        # Native pass(1,5): retain the eps input breakpoint and compare with
        # the source's explicitly split |x| reference using normest.
        f = cj.chebfun(lambda x: x, domain=(-1.0, EPS, 1.0)).abs()
        exact = cj.chebfun(lambda x: jnp.abs(x), domain=(-1.0, 0.0, 1.0))
        tol = 10.0 * EPS * exact.get("hscale")
        assert float((exact - f).normest()) < tol

        # Native pass(1,6): modulus of the unit-circle function is 1.
        circle = cj.chebfun(lambda x: jnp.exp(1j * x)).abs()
        tol = 10.0 * EPS * circle.get("hscale")
        assert float((circle - 1.0).normest()) < tol

    def test_native_real_abs_predicates(self):
        reference = _native_reference_f()
        tol = 20.0 * EPS * reference.get("hscale")
        bp = jnp.asarray(BREAKS)
        real = cj.chebfun(lambda x: jnp.sin(jnp.pi * x), domain=BREAKS)
        result = real.abs()
        assert len(result.funs) == 4
        assert float((reference - result).normest()) < tol
        assert bool(jnp.all(result.point_values ==
                            jnp.abs(jnp.sin(jnp.pi * bp))))
        direct = cj.chebfun(lambda x: jnp.abs(jnp.sin(jnp.pi * x)),
                            domain=BREAKS)
        assert len(direct.funs) == 4
        assert float((reference - direct).normest()) < tol
        assert bool(jnp.all(direct.point_values ==
                            jnp.abs(jnp.sin(jnp.pi * bp))))

    def test_native_imaginary_abs_predicates(self):
        reference = _native_reference_f()
        tol = 20.0 * EPS * reference.get("hscale")
        source = lambda x: 1j * jnp.cos(jnp.pi * (x - 0.5))
        imaginary = cj.chebfun(source, domain=BREAKS)
        result = imaginary.abs()
        assert len(result.funs) == 4
        assert float((reference - result).normest()) < tol
        assert result.isreal()
        direct = cj.chebfun(lambda x: jnp.abs(source(x)), domain=BREAKS)
        assert len(direct.funs) == 4
        assert float((reference - direct).normest()) < tol
        assert direct.isreal()

    def test_native_complex_abs_predicates(self):
        reference = jnp.sqrt(2.0) * _native_reference_f()
        tol = 20.0 * EPS * reference.get("hscale")
        source = lambda x: (jnp.sin(jnp.pi * x)
                            + 1j * jnp.cos(jnp.pi * (x - 0.5)))
        complex_value = cj.chebfun(source, domain=BREAKS)
        result = complex_value.abs()
        assert len(result.funs) == 4
        assert float((reference - result).normest()) < tol
        assert result.isreal()
        direct = cj.chebfun(lambda x: jnp.abs(source(x)), domain=BREAKS)
        assert len(direct.funs) == 4
        assert float((reference - direct).normest()) < tol
        assert direct.isreal()

    def test_array_valued(self):
        # Native pass(5,1:6): g4 compares against the existing f evaluated at
        # duplicated query columns; h4 is a separate direct abs callback.
        source_f = _native_reference_f()
        reference = cj.chebfun(
            lambda x: jnp.stack([source_f(x), source_f(x)], axis=-1),
            domain=BREAKS)
        sine = lambda x: jnp.sin(jnp.pi * x)
        cosine = lambda x: jnp.cos(jnp.pi * (x - 0.5))
        g = cj.chebfun(lambda x: jnp.stack([sine(x), cosine(x)], axis=-1),
                       domain=BREAKS)
        result = g.abs()
        assert len(result.funs) == 4
        tol = 20.0 * EPS * 2.0  # Native hscale on [-2,2].
        assert float((reference - result).normest()) < tol
        assert bool(jnp.all(result(X3) >= 0))
        direct = cj.chebfun(
            lambda x: jnp.abs(jnp.stack([sine(x), cosine(x)], axis=-1)),
            domain=BREAKS)
        assert len(direct.funs) == 4
        assert float((reference - direct).normest()) < tol
        assert bool(jnp.all(direct(X3) >= 0))
