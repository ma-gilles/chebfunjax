"""Port of MATLAB Chebfun tests/chebtech/test_extractBoundaryRoots.m (Opus 4.8).

The MATLAB file loops ``for n = 1:2`` over ``{chebtech1(), chebtech2()}`` and
factors out boundary roots (zeros at x = +/-1) of a chebtech via
``extractBoundaryRoots``.  chebfunjax now implements ``extractBoundaryRoots``
on both tech classes (coefficient-space peeling of the ``(1 +/- x)`` factors,
mirroring @chebtech/extractBoundaryRoots.m), so every assertion is exercised
at the SAME tolerance MATLAB uses.

Random test points are drawn in the OPEN interval ``(-1, 1)`` (as MATLAB does
via ``2*rand-1``); the analytic error checks hold at any such points.

Gap vs MATLAB (honest xfail):
- pass(n, 5) and pass(n, 6) build ``sin(1-x)/(1-x)``, which is a removable
  0/0 at x = +1.  Chebtech2 samples the endpoints and its constructor does
  not extrapolate non-finite endpoint values, so the *construction* yields
  NaN; Chebtech1 samples interior points and is unaffected.  These two cases
  are kept xfailed for Chebtech2 only, with a precise reason.

Provenance
----------
MATLAB source : tests/chebtech/test_extractBoundaryRoots.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

EPS = float(np.finfo(np.float64).eps)

# Deterministic test points in the open interval (-1, 1).
_RNG = np.random.default_rng(6178)
X = jnp.asarray(1.98 * _RNG.random(100) - 0.99)

_C2_ENDPOINT_NAN = (
    "sin(1-x)/(1-x) is a removable 0/0 at x=+1; Chebtech2 samples the "
    "endpoints and its constructor does not extrapolate non-finite endpoint "
    "values, so construction yields NaN (Chebtech1 samples interior points)"
)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechExtractBoundaryRoots:
    def test_left_endpoint_roots(self, Tech):
        # pass(n, 1): sin(2x)*(1+x)^3 -> multiplicity l == 3 at x = -1.
        ml = 3
        f = Tech.from_function(lambda x: jnp.sin(2 * x) * ((1 + x) ** ml))
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(lambda x: jnp.sin(2 * x))
        assert _ninf(g(X) - gexact(X)) < (1e3 ** ml) * EPS
        assert l == ml

    def test_right_endpoint_roots(self, Tech):
        # pass(n, 2): sin(cos(3x))*(1-x)^2 -> multiplicity r == 2 at x = 1.
        mr = 2
        f = Tech.from_function(lambda x: jnp.sin(jnp.cos(3 * x)) * ((1 - x) ** mr))
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(lambda x: jnp.sin(jnp.cos(3 * x)))
        assert _ninf(g(X) - gexact(X)) < (5e2 ** mr) * EPS
        assert r == mr

    def test_both_endpoint_roots(self, Tech):
        # pass(n, 3): exp(x)*(1+x)*(1-x)^2 -> l == 1, r == 2.
        ml, mr = 1, 2
        f = Tech.from_function(
            lambda x: jnp.exp(x) * ((1 + x) ** ml) * ((1 - x) ** mr)
        )
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(lambda x: jnp.exp(x))
        assert _ninf(g(X) - gexact(X)) < (1e2 ** (ml + mr)) * EPS
        assert l == ml and r == mr

    def test_complex_case(self, Tech):
        # pass(n, 4): complex integrand with boundary roots.
        ml, mr = 1, 2

        def f_op(x):
            return (x ** 2 + jnp.exp(x) + 1j * jnp.cos(2 * x)) * (
                (1 + x) ** ml
            ) * ((1 - x) ** mr)

        f = Tech.from_function(f_op)
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(
            lambda x: x ** 2 + jnp.exp(x) + 1j * jnp.cos(2 * x)
        )
        assert _ninf(g(X) - gexact(X)) < (2e1 ** (ml + mr)) * EPS
        assert l == ml and r == mr

    def test_no_roots(self, Tech):
        # pass(n, 5): sin(1-x)/(1-x) has no boundary roots (l == r == 0).
        if Tech is Chebtech2:
            pytest.xfail(_C2_ENDPOINT_NAN)
        f = Tech.from_function(lambda x: jnp.sin(1 - x) / (1 - x))
        g, l, r = f.extractBoundaryRoots()
        assert _ninf(g(X) - f(X)) == 0.0
        assert l == 0 and r == 0

    def test_roots_not_explicit(self, Tech):
        # pass(n, 6): sin(1-x) has an implicit root at x = 1 (r == 1).
        if Tech is Chebtech2:
            pytest.xfail(_C2_ENDPOINT_NAN)
        f = Tech.from_function(lambda x: jnp.sin(1 - x))
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(lambda x: jnp.sin(1 - x) / (1 - x))
        assert _ninf(g(X) - gexact(X)) < 1e2 * EPS
        assert r == 1

    def test_array_valued(self, Tech):
        # pass(n, 7): array-valued boundary root extraction.
        def f_op(x):
            return jnp.stack(
                [
                    jnp.sin(x) * ((1 - x) ** 2),
                    jnp.cos(x ** 2) * (1 + x) * (1 - x),
                ],
                axis=-1,
            )

        f = Tech.from_function(f_op)
        g, l, r = f.extractBoundaryRoots()
        gexact = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), jnp.cos(x ** 2)], axis=-1)
        )
        assert _ninf(g(X) - gexact(X)) < (1e2 ** 2) * EPS
        assert np.all(np.asarray(l) == [0, 1])
        assert np.all(np.asarray(r) == [2, 1])

    def test_full_arguments(self, Tech):
        # pass(n, 8): extractBoundaryRoots(f, [ml; mr]) with supplied mults.
        ml, mr = 1, 2
        f = Tech.from_function(
            lambda x: jnp.exp(x) * ((1 + x) ** ml) * ((1 - x) ** mr)
        )
        g, l, r = f.extractBoundaryRoots(num_roots=[[ml], [mr]])
        gexact = Tech.from_function(lambda x: jnp.exp(x))
        assert _ninf(g(X) - gexact(X)) < (1e2 ** (ml + mr)) * EPS

    def test_wrong_multiplicities(self, Tech):
        # pass(n, 9): supplied multiplicities exceed the true ones.
        ml, mr = 1, 2
        f = Tech.from_function(
            lambda x: jnp.exp(x) * ((1 + x) ** ml) * ((1 - x) ** mr)
        )
        g, l, r = f.extractBoundaryRoots(num_roots=[[ml + 1], [mr + 2]])
        gexact = Tech.from_function(lambda x: jnp.exp(x))
        assert _ninf(g(X) - gexact(X)) < 1e1 * (1e1 ** (ml + mr)) * EPS
        assert l == ml and r == mr
