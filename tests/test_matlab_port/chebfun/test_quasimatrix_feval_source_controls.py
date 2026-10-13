"""Source-shaped evaluation checks for Quasimatrix.__call__.

The two columns are explicit piecewise polynomials with different breaks,
so their values and one-sided limits are independently predictable.
"""

import jax.numpy as jnp
import numpy as np

import chebfunjax as cj
from chebfunjax.chebfun1d.linalg import Quasimatrix


def _quasimatrix():
    f1 = cj.chebfun(
        [lambda x: x**2, lambda x: x**2 + 1.0],
        domain=(-1.0, -0.25, 1.0),
    )
    f2 = cj.chebfun(
        [lambda x: 2.0 * x, lambda x: 2.0 * x - 3.0],
        domain=(-1.0, 0.35, 1.0),
    )
    return Quasimatrix([f1, f2], f1.domain)


def _exact(x):
    x = np.asarray(x)
    first = np.where(x < -0.25, x**2, x**2 + 1.0)
    second = np.where(x < 0.35, 2.0 * x, 2.0 * x - 3.0)
    if x.ndim < 2:
        return np.stack((first, second), axis=-1)
    return np.concatenate((first, second), axis=1)


def _assert_control_bound(got, expected):
    """Supplemental analytic control, not a native test_feval predicate."""
    got = np.asarray(got)
    expected = np.asarray(expected)
    bound = 20 * np.finfo(np.float64).eps * max(
        1.0, float(np.max(np.abs(expected)))
    )
    assert np.max(np.abs(got - expected), initial=0.0) < bound


def test_quasimatrix_matrix_query_packs_columns_along_second_axis():
    q = _quasimatrix()
    x = np.array([[-0.75, 0.1, 0.6], [-0.1, 0.8, -0.6]])
    got = np.asarray(q(jnp.asarray(x)))
    assert got.shape == (2, 6)
    _assert_control_bound(got, _exact(x))


def test_transposed_quasimatrix_matrix_query_swaps_first_two_axes_once():
    q = _quasimatrix().T
    x = np.array([[-0.75, 0.1, 0.6], [-0.1, 0.8, -0.6]])
    got = np.asarray(q(jnp.asarray(x)))
    assert got.shape == (6, 2)
    _assert_control_bound(got, _exact(x).T)


def test_quasimatrix_rank_one_python_adapter_and_empty_shape():
    q = _quasimatrix().T
    x = np.array([-0.75, 0.1, 0.6])
    got = np.asarray(q(jnp.asarray(x)))
    assert got.shape == (2, 3)
    _assert_control_bound(got, _exact(x).T)
    empty = np.asarray(q(jnp.zeros((0, 3))))
    assert empty.shape == (0, 3)


def test_quasimatrix_endpoint_aliases_and_one_sided_values():
    q = _quasimatrix()
    _assert_control_bound(q("left"), [1.0, -2.0])
    _assert_control_bound(q("right"), [2.0, -1.0])
    x = jnp.asarray([[-0.25], [0.35]])
    left = np.asarray(q(x, side="left"))
    right = np.asarray(q(x, side="right"))
    assert left.shape == right.shape == (2, 2)
    _assert_control_bound(left, [[0.0625, -0.5], [1.1225, 0.7]])
    _assert_control_bound(right, [[1.0625, -0.5], [1.1225, -2.3]])
    row_left = np.asarray(q.T("left"))
    assert row_left.shape == (2, 1)
    _assert_control_bound(row_left[:, 0], [1.0, -2.0])


def test_rank_three_query_packs_axis_two_and_row_transposes_once():
    q = _quasimatrix()
    x = np.linspace(-0.9, 0.9, 24).reshape((2, 3, 4))
    expected = _exact(x)
    got = np.asarray(q(jnp.asarray(x)))
    assert got.shape == (2, 6, 4)
    _assert_control_bound(got, expected)
    row = np.asarray(q.T(jnp.asarray(x)))
    assert row.shape == (6, 2, 4)
    _assert_control_bound(row, np.swapaxes(expected, 0, 1))


def test_quasimatrix_isempty_for_single_and_multiple_columns():
    q = _quasimatrix()
    assert not q.isempty()
    assert not Quasimatrix([q.cols[0]], q.domain).isempty()
