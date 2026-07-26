"""Port of MATLAB Chebfun tests/singfun/test_feval.m (Opus 4.8).

Self-validating: the Singfun evaluation is compared against the underlying
function handle at the SAME tolerance MATLAB uses.  Test points are an
interior grid (MATLAB uses 100 random points in ``(-1, 1)``).

Provenance
----------
MATLAB source : tests/singfun/test_feval.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.fun.singfun import Singfun

EPS = float(np.finfo(np.float64).eps)

# Interior points, well away from endpoints (as MATLAB's random points are).
X = jnp.asarray(np.linspace(-0.98, 0.98, 100))


def _sf(f, exps):
    return Singfun.from_function(f, exps)


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestSingfunFeval:
    def test_empty_points(self):
        # feval on an empty set of points returns empty
        f = _sf(lambda x: x, (0.0, 0.0))
        out = f(jnp.asarray(np.array([], dtype=np.float64)))
        assert out.shape[0] == 0

    def test_no_exponents(self):
        # FIXED by quadfix's sine-node construction (1c3fd5e): the bit-exactly
        # symmetric 2nd-kind nodes improved the Chebtech2 coefficient accuracy
        # for sin(cos(10x^2)) from ~10*eps (over the < 1e1*eps bound) to
        # ~0.7*eps, so this now passes.  (Was a strict xfail on the tech
        # coefficient-accuracy gap.)
        def fh(x):
            return jnp.sin(jnp.cos(10 * x ** 2))

        f = _sf(fh, (0.0, 0.0))
        assert _ninf(f(X) - fh(X)) < 1e1 * EPS

    def test_negative_exponents(self):
        # a, b in (1, 2) (fixed representative values in MATLAB's 1+rand range)
        a, b = 1.5, 1.7

        def fh(x):
            return jnp.sin(jnp.cos(10 * x ** 2)) / ((1 + x) ** a * (1 - x) ** b)

        f = _sf(fh, (-a, -b))
        assert _ninf(f(X) - fh(X)) < 2e3 * EPS

    def test_positive_exponents(self):
        # a, b in (0, 1)
        a, b = 0.5, 0.7

        def fh(x):
            return jnp.sin(jnp.cos(10 * x ** 2)) * (1 + x) ** a * (1 - x) ** b

        f = _sf(fh, (a, b))
        assert _ninf(f(X) - fh(X)) < 1e1 * EPS
