"""Port of MATLAB Chebfun tests/trigtech/test_any.m (Fable 5).

chebfunjax has no ``any()`` method, but MATLAB ``@trigtech/any.m`` is a plain
reduction over the physical-space values:

- ``any(f)``    (dim 1, down columns): ``any(f.values)`` -- per-column, is any
  value nonzero -> a 1 x m logical row.
- ``any(f, 2)`` (dim 2, across rows): evaluate at one arbitrary point and take
  ``any`` across the columns -> a scalar.

These tests build genuine array-valued (n, m) trigtechs and assert those
equivalents.  The empty-class case ``~any(trigtech())`` has no chebfunjax
analogue (no empty tech) and stays xfail.

Provenance
----------
MATLAB source : tests/trigtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech

# MATLAB's arbitrary evaluation point for any(f, 2).
_ARB_POINT = 0.1273881594


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechAny:
    @pytest.mark.xfail(
        reason="chebfunjax has no empty trigtech; the ~any(trigtech()) empty-class case "
        "has no analogue"
    )
    def test_empty(self):
        raise AssertionError("empty trigtech has no analogue")

    def test_columns(self):
        # pass(2): any(make(@(x) [sin(pi x) 0*x cos(pi x)])) == [1 0 1]
        # FIXED (Fable 5, Big-Three array-valued epic): any() over (n, m) values.
        f = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), 0 * x, jnp.cos(jnp.pi * x)], axis=-1))
        a = jnp.any(f.values != 0, axis=0)
        assert list(np.asarray(a).astype(int)) == [1, 0, 1]

    def test_rows(self):
        # pass(3): any(f, 2).coeffs == 1 for f = [sin(pi x) 0*x cos(pi x)]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), 0 * x, jnp.cos(jnp.pi * x)], axis=-1))
        x0 = jnp.array([_ARB_POINT], dtype=jnp.float64)
        assert int(jnp.any(f(x0)[0] != 0)) == 1

    def test_rows_zero(self):
        # pass(4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([0 * x, 0 * x], axis=-1))
        x0 = jnp.array([_ARB_POINT], dtype=jnp.float64)
        assert int(jnp.any(f(x0)[0] != 0)) == 0
