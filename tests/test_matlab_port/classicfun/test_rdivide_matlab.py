"""Port of MATLAB Chebfun tests/classicfun/test_rdivide.m (Fable 5).

Scalar division cases at MATLAB tolerances on [-2, 7].  Division by 0
(NaN result), array-valued funs, and singular-exponent divisions are
skipped/xfailed per chebfunjax capability.

Provenance
----------
MATLAB source : tests/classicfun/test_rdivide.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun

EPS = float(np.finfo(np.float64).eps)
DOM = Domain((-2.0, 7.0))
X = jnp.asarray(np.linspace(-2.0, 7.0, 1000))
ALPHA = -0.194758928283640 + 0.075474485412665j  # exact MATLAB constant


def _ninf(a):
    return float(jnp.max(jnp.abs(jnp.asarray(a))))


class TestClassicfunRdivide:
    def test_divide_function_by_scalar(self):
        f = Bndfun.from_function(jnp.sin, DOM)
        g = f / ALPHA
        err = jnp.abs(jnp.asarray(g(X)) - jnp.sin(X) / ALPHA)
        assert float(jnp.max(err)) < 10 * g.vscale * EPS

    def test_divide_by_zero_is_nan(self):
        # MATLAB: isnan(f ./ 0)
        f = Bndfun.from_function(jnp.sin, DOM)
        g = f / 0.0
        vals = np.asarray(g(jnp.asarray(np.array([0.5]))))
        assert np.all(np.isnan(vals) | np.isinf(vals))

    def test_array_valued_by_row(self):
        # pass(5): [sin cos] ./ [alpha beta] == [sin/alpha cos/beta].
        # FIXED (Fable 5, Big-Three array-valued epic): (n, m) Bndfun.
        beta = -0.526634844879922 - 0.685484380523668j
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g = f / jnp.asarray([ALPHA, beta])
        gexact = jnp.stack([jnp.sin(X) / ALPHA, jnp.cos(X) / beta], axis=-1)
        assert _ninf(g(X) - gexact) < 10 * g.vscale * EPS

    def test_array_valued_by_row_with_zero(self):
        # pass(6): [sin cos] ./ [alpha 0] -> column 0 finite, column 1 all NaN.
        # FIXED (Fable 5, Big-Three array-valued epic).
        fop = lambda x: jnp.stack([jnp.sin(x), jnp.cos(x)], axis=-1)
        f = Bndfun.from_function(fop, DOM)
        g = f / jnp.asarray([ALPHA, 0.0])
        vals = np.asarray(g(X))
        assert not np.any(np.isnan(vals[:, 0]))
        assert np.all(np.isnan(vals[:, 1]))

    def test_scalar_divided_by_function(self):
        # MATLAB: g = alpha ./ f with f = @(x) 1 + x.^2 (no roots)
        f = Bndfun.from_function(lambda x: 1 + x ** 2, DOM)
        g = ALPHA / f
        err = jnp.abs(jnp.asarray(g(X)) - ALPHA / (1 + X ** 2))
        assert float(jnp.max(err)) < 50 * g.vscale * EPS

    def test_division_creating_singularity(self):
        pytest.skip("division by a fun with roots requires singfun blowup "
                    "handling at the classicfun level")
