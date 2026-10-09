"""Native post-fixed numeric-empty population, pin7574c77.

@trigtech/trigtech.m136–157, populate.m36–55, vals2coeffs.m42–48:
fixed zero calls the actual operator at [1], then retains the empty result
shape/storage through population. This is distinct from a null operand.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech._trig_constructor import construct
from chebfunjax.tech.trigtech import Trigtech


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
@pytest.mark.parametrize('columns', [1, 3])
def test_fixed_zero_actual_callback_shape_storage_and_empty_mask(dtype, columns):
    calls = []
    def op(x):
        calls.append(x)
        value = 1 + 2j if dtype == jnp.complex128 else 1.
        return jnp.full((x.size, columns), value, dtype=dtype)
    f = Trigtech.from_function(op, n=0)
    assert len(calls) == 1 and jnp.array_equal(calls[0], jnp.asarray([1.]))
    assert f.coeffs.shape == f.values.shape == (0, columns)
    assert f.coeffs.dtype == f.values.dtype == dtype
    assert f.real_columns == () and f.isReal.size == 0 and f.is_real
    assert f.ishappy and f.vscale == 0


@pytest.mark.parametrize('dtype', [jnp.float64, jnp.complex128])
def test_empty_coefficient_population_is_not_null_operand(dtype):
    supplied = jnp.empty((0, 3), dtype=dtype)
    f = construct(supplied, coefficients=True, pref={'fixedLength': 0})
    assert f.coeffs.shape == f.values.shape == supplied.shape
    assert f.coeffs.dtype == f.values.dtype == supplied.dtype
    assert f.real_columns == () and f.ishappy


def test_empty_column_cache_survives_pytree_roundtrip_and_jit():
    f = Trigtech.from_function(lambda x: jnp.ones((x.size, 3), dtype=jnp.complex128), n=0)
    leaves, tree = jax.tree_util.tree_flatten(f)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    coeffs, values = jax.jit(lambda g: (g.coeffs, g.values))(restored)
    assert coeffs.shape == values.shape == (0, 3)
    assert coeffs.dtype == values.dtype == jnp.complex128
    assert restored.real_columns == () and restored.ishappy
