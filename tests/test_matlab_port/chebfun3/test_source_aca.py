"""Literal ACA ordering controls from @chebfun3/chebfun3f.m (7574c77)."""

import jax.numpy as jnp
import numpy as np

from chebfunjax.chebfun3d._chebfun3f_source import aca, decode_columns, unfold


def test_native_first_maximum_column_order():
    matrix = jnp.asarray([[0., 1.], [1., 0.]])
    *_, rows, columns = aca(matrix, 1e-15, 2)
    np.testing.assert_array_equal(rows, [1, 0])
    np.testing.assert_array_equal(columns, [0, 1])


def test_native_unique_maximum_and_strict_threshold():
    matrix = jnp.asarray([[0., 2.], [1., 0.]])
    *_, rows, columns = aca(matrix, 2., 2)
    np.testing.assert_array_equal(rows, [0])
    np.testing.assert_array_equal(columns, [1])


def test_native_rank_one_cancellation_and_complex_transpose():
    matrix = jnp.asarray([[1., 2.], [2., 4.]])
    ac, ar, at, rows, columns = aca(matrix, 1e-15, 2)
    np.testing.assert_array_equal(rows, [1])
    np.testing.assert_array_equal(columns, [1])
    np.testing.assert_array_equal(ac @ (ar.T / at[0, 0]), matrix)
    complex_matrix = matrix * 1j
    _, ar, _, rows, _ = aca(complex_matrix, 1e-15, 2)
    np.testing.assert_array_equal(ar, np.conj(np.asarray(complex_matrix)[[1], :].T))


def test_native_empty_matrix():
    ac, ar, at, rows, columns = aca(jnp.zeros((3, 0)), 1e-15, 3)
    assert (ac.shape, ar.shape, at.shape, rows.shape, columns.shape) == (
        (3, 0), (0, 0), (0, 0), (0,), (0,))


def test_native_unfold_and_paired_indices():
    tensor = jnp.asarray([[[100*i + 10*j + k for k in range(4)]
                           for j in range(3)] for i in range(2)])
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        matrix = unfold(tensor, axis)
        first, second = decode_columns(jnp.arange(matrix.shape[1]), tensor.shape[other[0]])
        for column in range(matrix.shape[1]):
            for row in range(matrix.shape[0]):
                index = [0, 0, 0]
                index[axis] = row
                index[other[0]] = int(first[column])
                index[other[1]] = int(second[column])
                assert matrix[row, column] == tensor[tuple(index)]


def test_native_nested_sampling_keeps_old_rows(monkeypatch):
    from chebfunjax.chebfun3d import _chebfun3f_source as source

    checks = iter([False, False, False, True, True, True])
    monkeypatch.setattr(source, "envelope_happiness", lambda *args: next(checks))
    old = tuple(jnp.full((3, 1), value) for value in (101., 202., 303.))
    pairs = tuple((jnp.asarray([1]), jnp.asarray([1])) for _ in range(3))
    calls = []

    def op(x, y, z):
        calls.append((x, y, z))
        return x + 2*y + 3*z

    result = source.refine_c1_fibers(op, old, pairs, (3, 3, 3),
                                    (-1., 1., -1., 1., -1., 1.), 1e-12)
    for axis in range(3):
        np.testing.assert_array_equal(result[axis][::2], old[axis])
        np.testing.assert_array_equal(result[axis][1::2], op(*calls[axis]))
        np.testing.assert_array_equal(calls[axis][axis][:, 0], result[axis+3][1::2])
    assert all(values.shape == (5, 1) for values in result[:3])


def test_source_inverse_tensor_product_and_deim():
    from chebfunjax.chebfun3d._chebfun3f_source import deim, invtprod

    tensor = jnp.arange(24., dtype=jnp.float64).reshape(2, 3, 4)
    result = invtprod(tensor, jnp.diag(jnp.asarray([1., 2.])),
                      jnp.eye(3)*4, jnp.eye(4)*8)
    np.testing.assert_array_equal(result[0], tensor[0]/32)
    np.testing.assert_array_equal(result[1], tensor[1]/64)
    basis = jnp.asarray([[0., 1.], [1., 0.], [0., 0.]])
    indices, selected = deim(basis)
    np.testing.assert_array_equal(indices, [1, 0])
    np.testing.assert_array_equal(selected, jnp.eye(2))
