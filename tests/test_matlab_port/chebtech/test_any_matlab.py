"""Port of MATLAB Chebfun tests/chebtech/test_any.m (Fable 5).

All four MATLAB assertions are ported, on genuine array-valued ``(n, m)``
techs and on the empty tech ``Tech.empty()``.

chebfunjax exposes no ``any()`` *method*, but MATLAB ``@chebtech/any.m`` is
a plain reduction over the representation, reproduced here directly:

- ``any(f)``    (dim 1, down columns): ``any(f.coeffs)`` -- per column, is any
  coefficient nonzero -> a 1 x m logical row.
- ``any(f, 2)`` (dim 2, across rows): evaluate at the arbitrary point
  ``0.1273881594`` and take ``any`` across the columns, storing the result as
  the single coefficient of a constant tech.
- ``any(emptyTech)`` is false, i.e. the empty tech holds no nonzero data.

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


def _any_dim2(Tech, f):
    """MATLAB ``any(f, 2)``: a constant tech holding the row-wise `any`."""
    vals = f(jnp.array([_ARB_POINT], dtype=jnp.float64))
    a = float(jnp.any(vals != 0))
    return Tech.from_coeffs(jnp.array([a], dtype=jnp.float64))


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechAny:
    def test_any_empty_class(self, Tech):
        # pass(n,1): ~any(testclass) -- the empty tech has no nonzero data
        f = Tech.empty()
        assert f.isempty()

    def test_any_down_columns(self, Tech):
        # pass(n,2): any(make(@(x) [sin(x) 0*x cos(x)])) == [1 0 1]
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        a = jnp.any(f.coeffs != 0, axis=0)
        assert list(np.asarray(a).astype(int)) == [1, 0, 1]

    def test_any_across_rows_nonzero(self, Tech):
        # pass(n,3): any(f, 2).coeffs == 1 for f = [sin(x) 0*x cos(x)]
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        g = _any_dim2(Tech, f)
        assert float(g.coeffs[0]) == 1.0

    def test_any_across_rows_zero(self, Tech):
        # pass(n,4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        f = Tech.from_function(lambda x: jnp.stack([0 * x, 0 * x], axis=-1))
        g = _any_dim2(Tech, f)
        assert float(g.coeffs[0]) == 0.0
