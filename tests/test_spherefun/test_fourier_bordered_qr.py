"""Independent small bordered-equation controls; not native sparse-LU identity."""

import numpy as np
import pytest

from chebfunjax.spherefun._bordered_qr import complex_givens, solve_bordered
from chebfunjax.spherefun._fourier_bands import build_source_operators

EPS = np.finfo(float).eps


def pack(a, lower, upper):
    n = len(a)
    ab = np.zeros((n, 2 * lower + upper + 1), dtype=a.dtype)
    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            ab[i, j - i + lower] = a[i, j]
    return ab


def qualify(ab, a, b, w, d, row, lower, upper):
    m = len(a)
    ii = np.delete(np.arange(m), row)
    full = np.vstack((w, a[ii]))
    right = np.concatenate((np.atleast_1d(d), b[ii])) if b.ndim == 1 else np.vstack((d, b[ii]))
    expected = np.linalg.solve(full, right)
    x, status = solve_bordered(ab, b, w, d, lower=lower, upper=upper, removed_row=row)
    assert not any(bool(flag) for flag in status)
    x = np.asarray(x)
    scale = np.linalg.norm(full, np.inf) * np.linalg.norm(x, np.inf) + np.linalg.norm(right, np.inf)
    assert np.linalg.norm(full @ x - right, np.inf) <= 64 * m * EPS * scale
    assert np.linalg.norm(x - expected, np.inf) <= (
        128 * m * EPS * np.linalg.cond(full, np.inf) * max(1, np.linalg.norm(expected, np.inf))
    )
    return x


@pytest.mark.parametrize("row", [0, 2, 4])
def test_diagonal_border_positions(row):
    a = np.diag(np.arange(1, 6, dtype=float))
    x = np.arange(1, 6, dtype=float) / 7
    w = np.arange(5, 0, -1, dtype=float)
    qualify(pack(a, 0, 0), a, a @ x, w, w @ x, row, 0, 0)


@pytest.mark.parametrize("complex_data", [False, True])
def test_asymmetric_band_multiple_rhs(complex_data):
    n, lower, upper = 7, 2, 1
    a = np.zeros((n, n), dtype=complex if complex_data else float)
    for i in range(n):
        for j in range(max(0, i - lower), min(n, i + upper + 1)):
            a[i, j] = (i + 2 * j + 1) / 17 + (1j * (i - j) / 11 if complex_data else 0)
    a += np.diag(5 + np.sum(abs(a), axis=1))
    w = a[3].copy() + np.arange(1, n + 1) / 31
    if complex_data:
        w += 1j * np.arange(n) / 29
    x = np.column_stack((np.arange(n) / 7 + 1, np.arange(n)[::-1] / 9))
    if complex_data:
        x = x + 1j * x[::-1] / 3
    qualify(pack(a, lower, upper), a, a @ x, w, w @ x, 3, lower, upper)


def test_singular_principal_minor_but_invertible_border():
    # Independent exact source-coefficient limit, NOT a production substitution.
    cs = np.array([0.25j, 0, 0, 0, -0.25j])
    sin2 = np.array([-0.25, 0, 0.5, 0, -0.25])
    ops = build_source_operators(4, 5, cs_coeffs=cs, sin2_coeffs=sin2)
    f = np.arange(20).reshape(4, 5) / 13 + 1j * np.arange(4)[:, None] / 7
    a, _, rhs, d = ops.helmholtz(f, 2, 1)
    dense = np.asarray(a.dense())
    ii = [0, 1, 3]
    assert np.linalg.det(dense[np.ix_(ii, ii)]) == 0
    qualify(
        a.ab,
        dense,
        np.asarray(rhs[:, 2]),
        np.asarray(ops.weights),
        np.asarray(d),
        2,
        a.lower,
        a.upper,
    )


@pytest.mark.parametrize("m,n,k", [(4, 5, 2), (8, 7, 1.3 + 0.4j), (16, 12, 2.3)])
def test_actual_helmholtz_border(m, n, k):
    ops = build_source_operators(m, n)
    f = np.arange(m * n).reshape(m, n) / 17 + 1j * np.arange(m)[:, None] / 13
    a, _, rhs, d = ops.helmholtz(f, k, 1)
    qualify(
        a.ab,
        np.asarray(a.dense()),
        np.asarray(rhs[:, n // 2]),
        np.asarray(ops.weights),
        np.asarray(d),
        m // 2,
        a.lower,
        a.upper,
    )


@pytest.mark.parametrize("m,n", [(5, 7), (8, 12)])
def test_actual_poisson_border(m, n):
    ops = build_source_operators(m, n)
    f = np.arange(m * n).reshape(m, n) / 19 + 1j * np.arange(m)[:, None] / 11
    rhs, _, d = ops.poisson_rhs(f)
    assert not np.any(np.asarray(ops.laplace.dense())[:, m // 2])
    qualify(
        ops.laplace.ab,
        np.asarray(ops.laplace.dense()),
        np.asarray(rhs[:, n // 2]),
        np.asarray(ops.weights),
        np.asarray(d),
        m // 2,
        ops.laplace.lower,
        ops.laplace.upper,
    )


def test_rank_and_constraint_statuses():
    zero = np.zeros((4, 4))
    _, status = solve_bordered(
        pack(zero, 1, 1), np.ones(4), np.ones(4), 1.0, lower=1, upper=1, removed_row=2
    )
    assert bool(status.rank_deficient)
    # C has full row rank; only its null direction is missed by the border.
    a = np.eye(4)
    w = np.array([1.0, 1.0, 0.0, 1.0])
    _, status = solve_bordered(pack(a, 1, 1), np.ones(4), w, 1.0, lower=1, upper=1, removed_row=2)
    assert not bool(status.rank_deficient)
    assert bool(status.constraint_singular)
    a[0, 0] = np.nan
    _, status = solve_bordered(
        pack(a, 1, 1), np.ones(4), np.ones(4), 1.0, lower=1, upper=1, removed_row=2
    )
    assert bool(status.nonfinite)


def test_complex_rotation_zero_phase_and_tiny_entries():
    for a, b in [(0j, 0j), (0j, 2 + 3j), (2 - 1j, 0j), (2 + 3j, -4 + 1j), (1 + 2j, 1e-18 - 2e-18j)]:
        c, s, r = complex_givens(np.asarray(a), np.asarray(b))
        c, s, r = float(c), complex(s), complex(r)
        g = np.array([[c, s], [-s.conjugate(), c]])
        assert np.linalg.norm(g @ np.array([a, b]) - np.array([r, 0]), np.inf) <= (
            32 * EPS * max(1, abs(a), abs(b))
        )
        assert np.linalg.norm(g.conj().T @ g - np.eye(2), np.inf) <= 32 * EPS
        if b != 0:
            assert s != 0


def test_two_row_complex_border():
    a = np.array([[2 + 1j, 1 - 1j], [1 + 2j, 3 - 1j]])
    w = np.array([2j, 1])
    x = np.array([1 - 1j, 2 + 0.3j])
    qualify(pack(a, 1, 1), a, a @ x, w, w @ x, 0, 1, 1)
