"""Supplemental shape/dtype controls for eager inverse-degree adapters."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

METHODS = ("acosd", "asind", "atand", "acot")


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.complex64))
def test_degree_adapter_preserves_supported_dtype_and_shape(method, dtype):
    if dtype is jnp.float32:
        values = jnp.asarray([0.25, -0.5], dtype=dtype)
    else:
        values = jnp.asarray([0.25 + 0.5j, -0.5 + 0.25j], dtype=dtype)
    result = np.asarray(_UNARY_FNS[method](values))
    assert result.shape == values.shape
    assert result.dtype == np.dtype(dtype)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_real_atan_preserves_jax_values_dtype_and_shape(dtype):
    # The source-qualified atan adapter now also preserves MATLAB's whole-array
    # complex-storage convention. Function-object identity would reject that
    # required adapter without testing a numerical or public API contract.
    values = jnp.asarray([[.25, -.5], [1., -2.]], dtype=dtype)
    result = _UNARY_FNS["atan"](values)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    np.testing.assert_array_equal(result, jnp.arctan(values))


@pytest.mark.parametrize("dtype,real_dtype", [
    (jnp.complex64, jnp.float32), (jnp.complex128, jnp.float64),
])
def test_atan_real_axis_complex_storage_matches_native(dtype, real_dtype):
    # Captured native whole-array storage: a zero imaginary output is stored
    # as real. Ordinary real-axis values must still agree with real atan.
    values = jnp.asarray([.25+0j, -.5+0j], dtype=dtype)
    result = _UNARY_FNS["atan"](values)
    assert result.shape == values.shape
    assert result.dtype == real_dtype
    np.testing.assert_array_equal(result, jnp.arctan(jnp.real(values)))
