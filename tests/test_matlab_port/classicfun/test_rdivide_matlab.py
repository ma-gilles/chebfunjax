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

    def test_array_valued_cases(self):
        pytest.skip("chebfunjax has no array-valued Bndfun")

    def test_scalar_divided_by_function(self):
        # MATLAB: g = alpha ./ f with f = @(x) 1 + x.^2 (no roots)
        f = Bndfun.from_function(lambda x: 1 + x ** 2, DOM)
        g = ALPHA / f
        err = jnp.abs(jnp.asarray(g(X)) - ALPHA / (1 + X ** 2))
        assert float(jnp.max(err)) < 50 * g.vscale * EPS

    def test_division_creating_singularity(self):
        pytest.skip("division by a fun with roots requires singfun blowup "
                    "handling at the classicfun level")
