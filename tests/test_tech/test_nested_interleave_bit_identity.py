"""Diagnostic bit identity for nested-grid data movement, not numerical bounds."""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.chebtech import _interleave_chebtech2_nested_values


@pytest.mark.parametrize(
    "old_dtype,new_dtype,columns",
    [
        (np.float64, np.float64, 1),
        (np.complex128, np.complex128, 1),
        (np.float64, np.float64, 3),
        (np.complex128, np.complex128, 3),
        (np.float32, np.float64, 2),
        (np.float64, np.complex128, 2),
        (np.complex64, np.complex128, 2),
    ],
)
def test_exact_interleaving(old_dtype, new_dtype, columns):
    shape = (5,) if columns == 1 else (5, columns)
    newshape = (4,) if columns == 1 else (4, columns)
    old = np.arange(np.prod(shape), dtype=np.float64).reshape(shape).astype(old_dtype)
    new = (-np.arange(np.prod(newshape), dtype=np.float64)).reshape(newshape).astype(new_dtype)
    old.reshape(-1)[:4] = [np.nan, np.inf, -np.inf, -0.0]
    new.reshape(-1)[:4] = [-0.0, -np.inf, np.inf, np.nan]
    if np.issubdtype(old.dtype, np.complexfloating):
        old.imag.reshape(-1)[:4] = [np.nan, -np.inf, np.inf, -0.0]
    if np.issubdtype(new.dtype, np.complexfloating):
        new.imag.reshape(-1)[:4] = [-0.0, np.inf, -np.inf, np.nan]
    values = jnp.asarray(old)
    new_values = jnp.asarray(new)
    dtype = jnp.result_type(values, new_values)
    expected = np.empty((9,) + old.shape[1:], dtype=np.dtype(dtype))
    expected[::2] = old
    expected[1:-1:2] = new
    actual = np.asarray(_interleave_chebtech2_nested_values(values, new_values))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()
    # The old eager primitive sequence is a separate baseline, not an oracle.
    baseline = jnp.empty(expected.shape, dtype=dtype).at[::2].set(values).at[1:-1:2].set(new_values)
    assert actual.tobytes() == np.asarray(baseline).tobytes()


def test_nan_payloads_and_signed_zero_are_moved():
    old = np.array(
        [0x7FF8000000000042, 0x8000000000000000, 0x7FF0000000000000], dtype=np.uint64
    ).view(np.float64)
    new = np.array([0xFFF8000000000031, 0xFFF0000000000000], dtype=np.uint64).view(np.float64)
    expected = np.empty(5)
    expected[::2] = old
    expected[1:-1:2] = new
    actual = np.asarray(_interleave_chebtech2_nested_values(jnp.asarray(old), jnp.asarray(new)))
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("length,cap", [(1, 65537), (17, 20)])
def test_refinement_stops_without_callback_or_replacing_values(length, cap):
    from chebfunjax.tech.chebtech import _refine_chebtech2_nested

    old = jnp.arange(length, dtype=jnp.float64)

    def forbidden_callback(x):
        raise AssertionError("Stop branch must not evaluate new nodes")

    actual, stopped = _refine_chebtech2_nested(forbidden_callback, old, max_length=cap)
    assert stopped and actual is old
