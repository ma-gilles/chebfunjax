"""Port of MATLAB Chebfun tests/trigtech/test_any.m (Fable 5).

MATLAB ``@trigtech/any.m`` reduces physical-space values. The default branch
calls MATLAB ``any(f.values)`` without a dimension, so its first-nonsingleton
dimension rule also matters for one-row inputs. ``any(f, 2)`` samples at the
source arbitrary point and returns a constant Trigtech indicating whether any
column is nonzero there.

These tests build genuine array-valued (n, m) trigtechs and assert the source
results, including the no-argument empty constructor call.

Provenance
----------
MATLAB source : tests/trigtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech


def _tt(f):
    return Trigtech.from_function(f)


class TestTrigtechAny:
    def test_empty(self):
        # pass(1): ~any(testclass), where testclass is trigtech().
        assert bool(Trigtech().any()) is False

    def test_columns(self):
        # pass(2): any(make(@(x) [sin(pi x) 0*x cos(pi x)])) == [1 0 1]
        # FIXED (Fable 5, Big-Three array-valued epic): any() over (n, m) values.
        f = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), 0 * x, jnp.cos(jnp.pi * x)], axis=-1))
        a = f.any()
        assert list(np.asarray(a).astype(int)) == [1, 0, 1]

    def test_rows(self):
        # pass(3): any(f, 2).coeffs == 1 for f = [sin(pi x) 0*x cos(pi x)]
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([jnp.sin(jnp.pi * x), 0 * x, jnp.cos(jnp.pi * x)], axis=-1))
        np.testing.assert_array_equal(np.asarray(f.any(dim=2).coeffs), np.array([[True]]))

    def test_rows_zero(self):
        # pass(4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        # FIXED (Fable 5, Big-Three array-valued epic).
        f = _tt(lambda x: jnp.stack([0 * x, 0 * x], axis=-1))
        np.testing.assert_array_equal(np.asarray(f.any(dim=2).coeffs), np.array([[False]]))
