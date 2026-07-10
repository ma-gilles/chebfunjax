"""Port of MATLAB Chebfun tests/chebtech/test_compose.m (Opus 4.8).

Self-validating: each composition is checked against the analytic composite
at the SAME tolerance MATLAB uses (multiples of vscale(h)*eps).  The MATLAB
file loops ``for n = 1:4``: n=1 is chebtech1(); n=2,3,4 are all chebtech2()
with only preference variants (n=3 sets ``refinementFunction='resampling'``,
n=4 sets ``extrapolate=true``).  chebfunjax has no such refinement/extrapolate
preferences, so n=3 and n=4 collapse into the single Chebtech2 parametrization
(their assertions are otherwise identical to n=2).

``compose`` exists ONLY on Chebtech2 in chebfunjax (Chebtech1 lacks it), so
every method xfails the Chebtech1 parametrization with a precise reason.
MATLAB mapping:
- ``compose(f, @sin)``          -> ``f.compose(jnp.sin)``          (op(f(x)))
- ``compose(f1, @plus, f2)``    -> ``f1.compose(op, f2)``          (op(f(x), g(x)))
- ``compose(f, g)`` (g a tech)  -> ``f.compose(g)``               (g(f(x)))

Gaps vs MATLAB (honest xfail/skip):
- Chebtech1 has no ``compose``.
- array-valued composition and the error-identifier cases: chebfunjax Chebtech
  is scalar-valued.

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

EPS = float(np.finfo(np.float64).eps)
# Dense deterministic test grid (analytic checks hold at any point).
X = jnp.asarray(np.linspace(-1.0, 1.0, 200))


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechCompose:
    def _skip_c1(self, Tech):
        if Tech is Chebtech1:
            pytest.xfail("Chebtech1 lacks .compose (Chebtech2-only method)")

    def test_compose_scalar_sin(self, Tech):
        # pass(n, 1): compose(f=x, @sin) has the coeffs of sin.
        self._skip_c1(Tech)
        f = Chebtech2.from_function(lambda x: x)
        g = f.compose(jnp.sin)
        h = Chebtech2.from_function(jnp.sin)
        n = max(g.n, h.n)
        gc = np.zeros(n)
        gc[: g.n] = np.asarray(g.coeffs)
        hc = np.zeros(n)
        hc[: h.n] = np.asarray(h.coeffs)
        assert _ninf(gc - hc) < 10 * h.vscale * EPS

    def test_compose_array_sin_2col(self, Tech):
        # pass(n, 2): compose([x x], @sin) (array-valued).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_array_sin_x_x2(self, Tech):
        # pass(n, 3): compose([x x^2], @sin) (array-valued).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_array_sin_3col(self, Tech):
        # pass(n, 4): compose([x x x^2], @sin) (array-valued).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_binary_plus(self, Tech):
        # pass(n, 5): compose(sin, @plus, cos) == sin + cos.
        self._skip_c1(Tech)
        f1 = Chebtech2.from_function(jnp.sin)
        f2 = Chebtech2.from_function(jnp.cos)
        g = f1.compose(lambda a, b: a + b, f2)
        ref = jnp.sin(X) + jnp.cos(X)
        vs = float(jnp.max(jnp.abs(ref)))
        assert _ninf(g(X) - ref) < 10 * vs * EPS

    def test_compose_binary_times_array(self, Tech):
        # pass(n, 6): compose([sin cos], @times, [cos exp]) (array-valued).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_gof_scalar(self, Tech):
        # pass(n, 7): compose(f=x^2, g=sin) == sin(x^2).
        self._skip_c1(Tech)
        f = Chebtech2.from_function(lambda x: x**2)
        g = Chebtech2.from_function(jnp.sin)
        h = f.compose(g)
        ref = jnp.sin(X**2)
        vs = float(jnp.max(jnp.abs(ref)))
        assert _ninf(h(X) - ref) < 10 * vs * EPS

    def test_compose_gof_array_g(self, Tech):
        # pass(n, 8): compose(f=x^2, g=[sin cos]) (array-valued g).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_gof_array_f(self, Tech):
        # pass(n, 9): compose(f=[x x^2], g=sin) (array-valued f).
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_arrval_error(self, Tech):
        # pass(n, 10): compose two array-valued techs -> error identifier.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )

    def test_compose_dim_error(self, Tech):
        # pass(n, 11): compose(f=[x x^2], @plus, g=sin) -> dimension error.
        pytest.skip(
            "chebfunjax Chebtech is scalar-valued; no array-valued techs"
        )
