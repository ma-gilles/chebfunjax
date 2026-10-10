"""Final ACA assembly block-boundary shape/order/dtype controls.

Native completeACA returns source-ordered pivots/rows/columns; this regression
checks only their storage assembly. Chebfun7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d._numeric_constructor import _stack_in_blocks


@pytest.mark.parametrize("count", [31, 32, 33, 64, 65])
@pytest.mark.parametrize("layout", ["pivots", "rows", "columns"])
@pytest.mark.parametrize("complex_values", [False, True])
def test_block_stack_preserves_shape_order_dtype(count, layout, complex_values):
    width = 1 if layout == "pivots" else 3
    raw = np.arange(count*width, dtype=np.float64).reshape(count, width)
    if complex_values:
        raw = raw.astype(np.complex128) + 1j*(raw + 1)
    items = [jnp.asarray(row[0] if layout == "pivots" else row) for row in raw]
    axis = 1 if layout == "columns" else 0
    expected = raw[:, 0] if layout == "pivots" else (raw.T if axis == 1 else raw)
    actual = np.asarray(_stack_in_blocks(items, axis=axis))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert np.array_equal(np.ascontiguousarray(actual).view(np.uint8), np.ascontiguousarray(expected).view(np.uint8))
