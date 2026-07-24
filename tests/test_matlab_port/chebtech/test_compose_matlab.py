"""Port of MATLAB Chebfun tests/chebtech/test_compose.m (Opus 4.8).

Self-validating: each composition is checked against the analytic composite
at the SAME tolerance MATLAB uses (multiples of vscale(h)*eps).  The MATLAB
file loops ``for n = 1:4``: n=1 is chebtech1(); n=2,3,4 are all chebtech2()
with only preference variants (n=3 sets ``refinementFunction='resampling'``,
n=4 sets ``extrapolate=true``).  chebfunjax has no such refinement/extrapolate
preferences, so n=3 and n=4 collapse into the single Chebtech2 parametrization
(their assertions are otherwise identical to n=2).

``compose`` now exists on BOTH tech classes in chebfunjax, so every method
is exercised on Chebtech1 and Chebtech2 (MATLAB's ``for n = 1:2`` loop).
MATLAB mapping:
- ``compose(f, @sin)``          -> ``f.compose(jnp.sin)``          (op(f(x)))
- ``compose(f1, @plus, f2)``    -> ``f1.compose(op, f2)``          (op(f(x), g(x)))
- ``compose(f, g)`` (g a tech)  -> ``f.compose(g)``               (g(f(x)))

Array-valued composition (pass 2, 3, 4, 6, 8, 9) is now supported: Chebtech
coefficients may be an (n, m) matrix (one function per column), and ``compose``
acts column-wise (FIXED, Fable 5, Big-Three array-valued epic).

Gaps vs MATLAB (honest xfail/skip):
- pass 11: binary compose of an array-valued f with a scalar-valued g does NOT
  broadcast in chebfunjax (raises on shape mismatch); modern MATLAB broadcasts.
  Kept skipped with a precise reason.

Provenance
----------
MATLAB source : tests/chebtech/test_compose.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
from chebfunjax.utils.quadrature import chebpts

EPS = float(np.finfo(np.float64).eps)
# Dense deterministic test grid (analytic checks hold at any point).
X = jnp.asarray(np.linspace(-1.0, 1.0, 200))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechCompose:
    def test_compose_scalar_sin(self, Tech):
        # pass(n, 1): compose(f=x, @sin) has the coeffs of sin.
        f = Tech.from_function(lambda x: x)
        g = f.compose(jnp.sin)
        h = Tech.from_function(jnp.sin)
        n = max(g.n, h.n)
        gc = np.zeros(n)
        gc[: g.n] = np.asarray(g.coeffs)
        hc = np.zeros(n)
        hc[: h.n] = np.asarray(h.coeffs)
        assert _ninf(gc - hc) < 10 * h.vscale * EPS

    def test_compose_array_sin_2col(self, Tech):
        # pass(n, 2): compose([x x], @sin) has the coeffs of [sin sin].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = Tech.from_function(lambda x: jnp.stack([x, x], axis=-1))
        g = f.compose(jnp.sin)
        h = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.sin(x)], axis=-1)
        )
        n = max(g.n, h.n)
        gc = np.zeros((n, g.coeffs.shape[1]))
        gc[: g.n] = np.asarray(g.coeffs)
        hc = np.zeros((n, h.coeffs.shape[1]))
        hc[: h.n] = np.asarray(h.coeffs)
        # MATLAB: 10*max(vscale(h)*eps) with vscale(h) the scalar global max.
        assert _ninf(gc - hc) < 10 * h.vscale * EPS

    def test_compose_array_sin_x_x2(self, Tech):
        # pass(n, 3): compose([x x^2], @sin) == sin([x x^2]) at the grid.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = Tech.from_function(lambda x: jnp.stack([x, x**2], axis=-1))
        g = f.compose(jnp.sin)
        # chebfunjax has no g.points(); the Chebtech2 grid is chebpts(g.n, 2).
        xc = chebpts(g.n, 2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        ref = jnp.stack([jnp.sin(xc), jnp.sin(xc**2)], axis=-1)
        assert _ninf(ref - values) < 1e2 * g.vscale * EPS

    def test_compose_array_sin_3col(self, Tech):
        # pass(n, 4): compose([x x x^2], @sin) == sin([x x x^2]) at the grid.
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f = Tech.from_function(lambda x: jnp.stack([x, x, x**2], axis=-1))
        g = f.compose(jnp.sin)
        xc = chebpts(g.n, 2)
        values = Chebtech2.coeffs2vals(g.coeffs)
        ref = jnp.stack([jnp.sin(xc), jnp.sin(xc), jnp.sin(xc**2)], axis=-1)
        assert _ninf(ref - values) < 1e2 * g.vscale * EPS

    def test_compose_binary_plus(self, Tech):
        # pass(n, 5): compose(sin, @plus, cos) == sin + cos.
        f1 = Tech.from_function(jnp.sin)
        f2 = Tech.from_function(jnp.cos)
        g = f1.compose(lambda a, b: a + b, f2)
        ref = jnp.sin(X) + jnp.cos(X)
        vs = float(jnp.max(jnp.abs(ref)))
        assert _ninf(g(X) - ref) < 10 * vs * EPS

    def test_compose_binary_times_array(self, Tech):
        # pass(n, 6): compose([sin cos], @times, [cos exp]) == [sin*cos cos*exp].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) coeffs.
        f1 = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        f2 = Tech.from_function(
            lambda x: jnp.stack([jnp.cos(x), jnp.exp(x)], axis=-1)
        )
        g = f1.compose(lambda a, b: a * b, f2)
        h = Tech.from_function(
            lambda x: jnp.stack(
                [jnp.sin(x) * jnp.cos(x), jnp.cos(x) * jnp.exp(x)], axis=-1
            )
        )
        hvalues = Chebtech2.coeffs2vals(h.coeffs)
        gvalues = Chebtech2.coeffs2vals(g.coeffs)
        # MATLAB: 10*max(10*vscale(h)*eps) == 100*vscale(h)*eps.
        assert _ninf(hvalues - gvalues) < 100 * h.vscale * EPS

    def test_compose_gof_scalar(self, Tech):
        # pass(n, 7): compose(f=x^2, g=sin) == sin(x^2).
        f = Tech.from_function(lambda x: x**2)
        g = Tech.from_function(jnp.sin)
        h = f.compose(g)
        ref = jnp.sin(X**2)
        vs = float(jnp.max(jnp.abs(ref)))
        assert _ninf(h(X) - ref) < 10 * vs * EPS

    def test_compose_gof_array_g(self, Tech):
        # pass(n, 8): compose(f=x^2, g=[sin cos]) == [sin(x^2) cos(x^2)].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued g.
        f = Tech.from_function(lambda x: x**2)
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        h = f.compose(g)
        xc = chebpts(h.n, 2)
        hvalues = Chebtech2.coeffs2vals(h.coeffs)
        ref = jnp.stack([jnp.sin(xc**2), jnp.cos(xc**2)], axis=-1)
        assert _ninf(hvalues - ref) < 10 * h.vscale * EPS

    def test_compose_gof_array_f(self, Tech):
        # pass(n, 9): compose(f=[x x^2], g=sin) == [sin(x) sin(x^2)].
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued f.
        f = Tech.from_function(lambda x: jnp.stack([x, x**2], axis=-1))
        g = Tech.from_function(jnp.sin)
        h = f.compose(g)
        xc = chebpts(h.n, 2)
        hvalues = Chebtech2.coeffs2vals(h.coeffs)
        ref = jnp.stack([jnp.sin(xc), jnp.sin(xc**2)], axis=-1)
        assert _ninf(hvalues - ref) < 10 * h.vscale * EPS

    def test_compose_arrval_error(self, Tech):
        # pass(n, 10): compose two array-valued techs f(g) is unsupported.
        # FIXED (Fable 5, Big-Three array-valued epic): array-valued techs now
        # exist, and composing two of them as g(f) is genuinely unsupported --
        # it raises. chebfunjax has no MATLAB 'CHEBFUN:CHEBTECH:compose:arrval'
        # error identifier, so we assert only that it raises.
        f = Tech.from_function(lambda x: jnp.stack([x, x**2], axis=-1))
        g = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        )
        with pytest.raises(Exception):
            f.compose(g)

    def test_compose_dim_error(self, Tech):
        # pass(n, 11): compose(f=[x x^2], @plus, g=sin).
        # FIXED (Fable 5, Big-Three array-valued epic): binary compose
        # now broadcasts a scalar-valued operand against an array-valued
        # one (modern MATLAB >= R2016b semantics -- the test's non-error
        # branch).
        f = Tech.from_function(
            lambda x: jnp.stack([x, x ** 2], axis=-1))
        g = Tech.from_function(lambda x: jnp.sin(x))
        h = f.compose(lambda a, b: a + b, g)
        xs = jnp.asarray(np.linspace(-1.0, 1.0, 30))
        exact = jnp.stack(
            [xs + jnp.sin(xs), xs ** 2 + jnp.sin(xs)], axis=-1)
        assert float(jnp.max(jnp.abs(h(xs) - exact))) \
            < 10 * h.vscale * EPS
