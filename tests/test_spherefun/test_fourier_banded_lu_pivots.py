import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.spherefun._banded_lu import solve_banded

from .test_fourier_banded_lu import pack, qualify


@pytest.mark.parametrize("complex_data", [False, True])
@pytest.mark.parametrize("n", [3, 7])
def test_distance_two_pivot_and_outer_fill(n, complex_data):
    a = (
        np.diag(4 * np.ones(n))
        + np.diag(0.25 * np.ones(n - 1), 1)
        + np.diag(0.5 * np.ones(n - 1), -1)
    )
    a += np.diag(0.3 * np.ones(n - 2), 2) + np.diag(0.2 * np.ones(n - 2), -2)
    a[0, 0] = 0
    a[1, 0] = 1
    a[2, 0] = 5
    if n > 4:
        a[2, 4] = 2
    if complex_data:
        a = a * (1 + 0.2j)
    # Independent first Gaussian step establishes the required path/fill.
    assert np.argmax(abs(a[:3, 0])) == 2
    work = a.copy()
    work[[0, 2]] = work[[2, 0]]
    if n > 4:
        assert work[0, 4] != 0
        work[1] -= (work[1, 0] / work[0, 0]) * work[0]
        assert work[1, 4] != 0
    qualify(a, 2, 2, matrix_rhs=True)


@pytest.mark.parametrize("n", [1, 2])
def test_storage_bandwidth_larger_than_system(n):
    a = np.diag(np.arange(1, n + 1, dtype=float))
    if n == 2:
        a[0, 0] = 0
        a[0, 1] = 1
        a[1, 0] = 2
    qualify(a, 2, 2)


@pytest.mark.parametrize("complex_data", [False, True])
def test_repeated_distance_two_pivots(complex_data):
    block = np.array([[0.0, 0.0, 1.0], [1.0, 2.0, 1.0], [3.0, 0.0, 4.0]])
    a = np.kron(np.eye(2), block)
    if complex_data:
        a = a * (1 + 2j)
    assert np.argmax(abs(a[:3, 0])) == 2
    assert np.argmax(abs(a[3:6, 3])) == 2
    qualify(a, 2, 2, matrix_rhs=True)


def test_trailing_rank_deficiency_flag():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    _, bad = solve_banded(jnp.asarray(pack(a, 1, 1)), jnp.ones(2), lower=1, upper=1)
    assert bool(bad)
