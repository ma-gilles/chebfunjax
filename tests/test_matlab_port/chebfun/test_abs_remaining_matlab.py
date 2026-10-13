"""Remaining clauses from MATLAB ``tests/chebfun/test_abs.m``.

The section 6 source ``quasi2cheb`` calls are identity operations: the inputs
are single array-valued Chebfun objects and native ``quasi2cheb`` returns
unchanged whenever ``numel(F) < 2``. The adapter below records that source
operation without constructing a genuine Quasimatrix.

Provenance
----------
MATLAB source : tests/chebfun/test_abs.m, sections 6--11
Chebfun commit: 7574c77
Native random data: fixtures/abs_remaining_seed6178_native.json (R2025b)
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj

EPS = float(np.finfo(np.float64).eps)
_FIXTURE = Path(__file__).with_name("fixtures") / "abs_remaining_seed6178_native.json"
with _FIXTURE.open(encoding="utf-8") as _stream:
    _NATIVE = json.load(_stream)
X_SINGULAR = jnp.asarray(_NATIVE["x_singular"], dtype=jnp.float64)
X_PIECE = jnp.asarray(_NATIVE["x_piece"], dtype=jnp.float64)
X_ENTIRE = jnp.asarray(_NATIVE["x_entire"], dtype=jnp.float64)
X_HALFLINE = jnp.asarray(_NATIVE["x_halfline"], dtype=jnp.float64)
with (Path(__file__).with_name("fixtures") / "abs_seed6178_native.json").open(
        encoding="utf-8") as _stream:
    _INITIAL_NATIVE = json.load(_stream)
X3 = jnp.asarray(_INITIAL_NATIVE["x3"], dtype=jnp.float64)


def _source_quasi2cheb_identity(f):
    """Mirror the native ``quasi2cheb(F)`` numel(F)<2 identity branch."""
    return f


class TestChebfunAbsRemaining:
    def test_native_array_quasi2cheb_identity_predicates(self):
        # Native pass(6,1:6). The source inputs are array-valued Chebfuns,
        # and quasi2cheb returns each object unchanged for numel(F)==1.
        sine = lambda x: jnp.sin(jnp.pi * x)
        cosine = lambda x: jnp.cos(jnp.pi * (x - 0.5))
        breaks = (-2.0, -1.0, 0.0, 1.0, 2.0)
        source_f = cj.chebfun(
            [sine, lambda x: -sine(x), sine, lambda x: -sine(x)],
            domain=breaks)
        reference = cj.chebfun(
            lambda x: jnp.stack([source_f(x), source_f(x)], axis=-1),
            domain=breaks)
        g = cj.chebfun(
            lambda x: jnp.stack([sine(x), cosine(x)], axis=-1),
            domain=breaks)
        g4 = _source_quasi2cheb_identity(g).abs()
        assert len(g4.funs) == 4
        tol = 20.0 * EPS * 2.0
        assert float((reference - g4).normest()) < tol
        assert bool(jnp.all(g4(X3) >= 0))

        direct = cj.chebfun(
            lambda x: jnp.abs(jnp.stack([sine(x), cosine(x)], axis=-1)),
            domain=breaks)
        h4 = _source_quasi2cheb_identity(direct)
        assert len(h4.funs) == 4
        assert float((reference - h4).normest()) < tol
        assert bool(jnp.all(h4(X3) >= 0))

    def test_native_complicated_complex_abs_overwritten_pass7(self):
        # The source later overwrites pass(7,:) with its singular predicate;
        # keep this earlier source assertion as its own test.
        source = lambda x: jnp.sin(1j * x) * (1j * x + jnp.exp(5j * x))
        f = cj.chebfun(source)
        g = cj.chebfun(lambda x: jnp.abs(source(x)), domain=(-1.0, 0.0, 1.0))
        h = f.abs()
        assert float((g - h).normest()) < 100.0 * EPS

    def test_native_singular_abs_overwritten_pass7(self):
        # This is the later assignment to pass(7,:); X_SINGULAR was captured
        # after seedRNG(6178), then the exact native constructor sequence.
        domain = (-2.0, 7.0)
        power = -1.64
        op = lambda x: (x - domain[0]) ** power
        f = cj.chebfun(op, domain=domain, exps=(power, 0.0))
        g = f.abs()
        values = g(X_SINGULAR)
        exact = jnp.abs(op(X_SINGULAR))
        assert float(jnp.linalg.norm(values - exact, ord=jnp.inf)) < (
            1.0e4 * EPS * float(jnp.linalg.norm(exact, ord=jnp.inf)))

    def test_native_piecewise_singular_abs(self):
        domain = (-1.0, 1.0)
        domain_check = (domain[0] + 0.1, domain[1] - 0.1)
        power1, power2 = -0.5, -1.2

        def op(x):
            # MATLAB promotes negative real bases to the principal complex
            # branch for a noninteger power. Make that dtype explicit in JAX.
            left = (x - domain[0]).astype(jnp.complex128) ** power1
            right = (x - domain[1]).astype(jnp.complex128) ** power2
            return jnp.sin(10.0 * x) * left * right

        f = cj.chebfun(op, domain=domain, exps=(power1, power2), splitting=True)
        g = f.abs()
        assert domain_check[0] < float(jnp.min(X_PIECE))
        assert float(jnp.max(X_PIECE)) < domain_check[1]
        values = g(X_PIECE)
        exact = op(X_PIECE)
        error = values - jnp.abs(exact)
        centered = error - jnp.mean(error)
        assert float(jnp.linalg.norm(centered, ord=jnp.inf)) < (
            100.0 * EPS * float(jnp.linalg.norm(exact, ord=jnp.inf)))

    def test_native_entire_unbounded_abs(self):
        op = lambda x: (1.0 - jnp.exp(-(x ** 2))) / x
        f = cj.chebfun(op, domain=(-jnp.inf, jnp.inf))
        g = f.abs()
        error = g(X_ENTIRE) - jnp.abs(op(X_ENTIRE))
        assert float(jnp.linalg.norm(error, ord=jnp.inf)) < (
            10.0 * EPS * g.vscale)

    def test_native_unbounded_blowup_abs(self):
        op = lambda x: -(x ** 2) * (1.0 + jnp.exp(-(x ** 2)))
        f = cj.chebfun(op, domain=(-jnp.inf, jnp.inf), exps=(2.0, 2.0))
        g = f.abs()
        error = g(X_ENTIRE) - jnp.abs(op(X_ENTIRE))
        assert float(jnp.linalg.norm(error, ord=jnp.inf)) < (
            1.0e5 * EPS * g.vscale)

    def test_native_halfline_unbounded_abs(self):
        op = lambda x: 0.75 + jnp.sin(10.0 * x) / jnp.exp(x)
        f = cj.chebfun(op, domain=(0.0, jnp.inf))
        g = f.abs()
        error = g(X_HALFLINE) - jnp.abs(op(X_HALFLINE))
        assert float(jnp.linalg.norm(error, ord=jnp.inf)) < (
            20.0 * EPS * g.vscale)
