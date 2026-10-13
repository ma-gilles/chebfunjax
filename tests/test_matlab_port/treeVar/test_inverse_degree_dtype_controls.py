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


def test_raw_atan_dispatch_remains_jax_primitive():
    assert _UNARY_FNS["atan"] is jnp.arctan
