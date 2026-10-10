"""Exact data-selection adapters for native sphere factor subsets (7574c77)."""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.spherefun._factor_assembly import (
    select_factor_columns,
    select_factor_entries,
)


def exact(actual, expected):
    a, b = jax.device_get(actual), jax.device_get(expected)
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()


def operation(matrix):
    return select_factor_columns if matrix else select_factor_entries


def eager(values, index, matrix):
    return values[:, index] if matrix else values[index]


def fixture(matrix, complex_data):
    data = [[1., -0., 3., 4., 5., 6., 7.],
            [8., 9., 10., 11., 12., 13., 14.]] if matrix else [1., -0., 3., 4., 5., 6., 7.]
    value = jnp.asarray(data, dtype=jnp.complex128 if complex_data else jnp.float64)
    if complex_data:
        value = value + 1j * jnp.arange(value.size).reshape(value.shape)
        assert bool(jnp.any(jnp.imag(value) != 0))
    return value


@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('indices', [[0, 1, 2], [6, 2, 4], [2, 2, 0], [-1, -7, -3], []])
def test_exact_index_order_dtype_and_eager_bits(matrix, complex_data, indices):
    values = fixture(matrix, complex_data)
    source = jax.device_get(values).tolist()
    reference = ([[row[i] for i in indices] for row in source] if matrix
                 else [source[i] for i in indices])
    expected = jnp.asarray(reference, dtype=values.dtype)
    for index_dtype in (jnp.int32, jnp.int64):
        index = jnp.asarray(indices, dtype=index_dtype)
        actual = operation(matrix)(values, index)
        exact(actual, eager(values, index, matrix))
        exact(actual, expected)


@pytest.mark.parametrize('matrix', [False, True])
def test_nonfinite_and_signed_component_copies(matrix):
    for complex_data in (False, True):
        data = [float('nan'), float('inf'), -float('inf'), -0., 0., 2.]
        if complex_data:
            data = [complex(x, y) for x, y in zip(data, [-0., 3., -4., 0., -0., float('nan')])]
        values = jnp.asarray([data, data[::-1]] if matrix else data)
        index = jnp.asarray([3, 0, -1, 2, 3], dtype=jnp.int32)
        exact(operation(matrix)(values, index), eager(values, index, matrix))


@pytest.mark.parametrize('matrix', [False, True])
@pytest.mark.parametrize('complex_data', [False, True])
def test_jvp_and_repeated_index_real_gradient(matrix, complex_data):
    values = fixture(matrix, complex_data)
    tangent = fixture(matrix, complex_data) * 2
    index = jnp.asarray([2, 2, -1, 0], dtype=jnp.int32)
    def fn(x):
        return operation(matrix)(x, index)
    primal, derivative = jax.jvp(jax.jit(fn), (values,), (tangent,))
    exact(primal, eager(values, index, matrix))
    exact(derivative, eager(tangent, index, matrix))
    if not complex_data:
        gradient = jax.grad(lambda x: jnp.sum(fn(x)))(values)
        counts = [1., 0., 2., 0., 0., 0., 1.]
        exact(gradient, jnp.asarray([counts, counts] if matrix else counts))
