"""Port of MATLAB Chebfun tests/chebtech/test_any.m (Fable 5).

chebfunjax has no ``any()`` method, but MATLAB ``@chebtech/any.m`` is a plain
reduction over the representation:

- ``any(f)``    (dim 1, down columns): ``any(f.coeffs)`` -- per-column, is any
  coefficient nonzero -> a 1 x m logical row.
- ``any(f, 2)`` (dim 2, across rows): evaluate at one arbitrary point and take
  ``any`` across the columns -> a scalar (stored as a constant tech).

These tests build genuine array-valued (n, m) techs and assert those
equivalents.  The empty-class case ``~any(chebtech1())`` has no chebfunjax
analogue (no empty tech) and stays skipped.

Provenance
----------
MATLAB source : tests/chebtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

# MATLAB's arbitrary evaluation point for any(f, 2).
_ARB_POINT = 0.1273881594


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechAny:
    def test_any_empty_class(self, Tech):
        # pass(n,1): ~any(testclass)
        pytest.skip(
            "chebfunjax has no empty tech; the ~any(chebtech()) empty-class "
            "case has no analogue"
        )

    def test_any_down_columns(self, Tech):
        # pass(n,2): any(make(@(x) [sin(x) 0*x cos(x)])) == [1 0 1]
        # FIXED (Fable 5, Big-Three array-valued epic): any() over (n, m) coeffs.
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        a = jnp.any(f.coeffs != 0, axis=0)
        assert list(np.asarray(a).astype(int)) == [1, 0, 1]

    def test_any_across_rows_nonzero(self, Tech):
        # pass(n,3): any(f, 2).coeffs == 1 for f = [sin(x) 0*x cos(x)]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        x0 = jnp.array([_ARB_POINT], dtype=jnp.float64)
        assert int(jnp.any(f(x0)[0] != 0)) == 1

    def test_any_across_rows_zero(self, Tech):
        # pass(n,4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = Tech.from_function(lambda x: jnp.stack([0 * x, 0 * x], axis=-1))
        x0 = jnp.array([_ARB_POINT], dtype=jnp.float64)
        assert int(jnp.any(f(x0)[0] != 0)) == 0
