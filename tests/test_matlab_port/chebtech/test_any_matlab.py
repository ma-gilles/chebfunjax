"""Port of MATLAB Chebfun tests/chebtech/test_any.m.

All four native assertions call the public ``Chebtech.any`` method directly.

Provenance
----------
MATLAB source : tests/chebtech/test_any.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
class TestChebtechAny:
    def test_any_empty_class(self, Tech):
        # pass(n,1): ~any(testclass) -- the empty tech has no nonzero data
        f = Tech.empty()
        assert not bool(f.any())

    def test_any_down_columns(self, Tech):
        # pass(n,2): any(make(@(x) [sin(x) 0*x cos(x)])) == [1 0 1]
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        a = f.any()
        assert list(np.asarray(a).astype(int)) == [1, 0, 1]

    def test_any_across_rows_nonzero(self, Tech):
        # pass(n,3): any(f, 2).coeffs == 1 for f = [sin(x) 0*x cos(x)]
        f = Tech.from_function(
            lambda x: jnp.stack([jnp.sin(x), 0 * x, jnp.cos(x)], axis=-1)
        )
        g = f.any(2)
        assert float(g.coeffs[0]) == 1.0

    def test_any_across_rows_zero(self, Tech):
        # pass(n,4): any(make(@(x) [0*x 0*x]), 2).coeffs == 0
        f = Tech.from_function(lambda x: jnp.stack([0 * x, 0 * x], axis=-1))
        g = f.any(2)
        assert float(g.coeffs[0]) == 0.0

    def test_any_invalid_dimension(self, Tech):
        f = Tech.from_coeffs(np.array([1.0]))
        with pytest.raises(ValueError, match="DIM input must be 1 or 2"):
            f.any(3)

    def test_any_jit(self, Tech):
        f = Tech.from_coeffs(jnp.array([[0.0, 2.0], [0.0, 0.0]]))
        result = jax.jit(lambda tech: tech.any())(f)
        np.testing.assert_array_equal(np.asarray(result), [False, True])


@pytest.mark.parametrize("Tech", [Chebtech1, Chebtech2])
def test_any_native_nan_and_first_nonsingleton(Tech):
    # Source any(f.coeffs) has no explicit dimension and ignores NaN.
    assert not Tech.from_coeffs(jnp.asarray([jnp.nan])).any()
    assert not Tech.from_coeffs(jnp.asarray([jnp.nan])).any(2).coeffs[0]
    row = Tech.from_coeffs(jnp.asarray([[0., jnp.nan, 2.]]))
    assert row.any().shape == () and bool(row.any())
    columns = Tech.from_coeffs(jnp.asarray([[0., jnp.nan, 2.], [0., 0., 0.]]))
    np.testing.assert_array_equal(columns.any(), [False, False, True])
    empty_columns = Tech(coeffs=jnp.empty((0, 3)))
    np.testing.assert_array_equal(empty_columns.any(), [False, False, False])
    unhappy = Tech(coeffs=jnp.asarray([1.]), ishappy=False)
    assert not unhappy.any(2).ishappy
