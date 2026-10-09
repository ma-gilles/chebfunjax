"""Independent bounded source operator equations; no solve or large data."""

import numpy as np
import pytest

from chebfunjax.spherefun._fourier_bands import (
    band_apply,
    build_source_operators,
    differentiation_diagonal,
    multiplier_bands,
)

EPS = np.finfo(float).eps


def matrix_bound(actual, expected, m):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert np.max(np.abs(actual - expected), initial=0) <= (
        64 * m * EPS * max(1, np.max(np.abs(expected), initial=0))
    )


def rhs_bound(actual, expected, m):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = (
        abs((actual - expected).item())
        if expected.ndim == 0
        else np.linalg.norm(actual - expected, np.inf)
    )
    scale = abs(expected.item()) if expected.ndim == 0 else np.linalg.norm(expected, np.inf)
    assert error <= 128 * m * EPS * max(1, scale)


@pytest.mark.parametrize("m,n", [(8, 7), (8, 12), (16, 7), (16, 12)])
def test_source_dense_equations(m, n):
    from chebfunjax.spherefun.spherefun import _sphere_fourier_operators

    d1, d2, dn, cs, s2, en, zero = _sphere_fourier_operators(m, n)
    bands = build_source_operators(m, n)
    assert bands.zero_latitude == zero
    assert bands.zero_longitude == n // 2
    matrix_bound(bands.dtheta1, np.diag(d1), m)
    matrix_bound(bands.dtheta2, np.diag(d2), m)
    matrix_bound(bands.dlambda2, np.diag(dn), m)
    matrix_bound(bands.cs.dense(), cs, m)
    matrix_bound(bands.sin2.dense(), s2, m)
    matrix_bound(bands.laplace.dense(), s2 @ d2 + cs @ d1, m)
    rhs_bound(bands.weights, en, m)
    # Tiny odd bands/weights must not disappear inside a generous absolute bound.
    assert np.array_equal(np.diag(np.asarray(bands.cs.dense()), 1), np.diag(cs, 1))
    assert np.array_equal(np.diag(np.asarray(bands.sin2.dense()), 1), np.diag(s2, 1))
    assert np.any(np.diag(np.asarray(bands.laplace.dense()), 1) != 0)
    modes = np.arange(m) - m // 2
    odd = (modes % 2 != 0) & (np.abs(modes) != 1)
    assert np.all(np.imag(np.asarray(bands.weights)[odd]) != 0)
    assert np.array_equal(np.asarray(bands.weights)[[zero - 1, zero + 1]], [0, 0])
    for operator in (bands.cs, bands.sin2, bands.laplace):
        assert (
            operator.lower
            == operator.upper
            == min(m - 1, max(len(bands.cs_coeffs) // 2, len(bands.sin2_coeffs) // 2))
        )
        assert not np.any(np.asarray(operator.ab)[:, operator.lower + operator.upper + 1 :])
    f = (np.arange(m * n).reshape(m, n) + 1) / 17 + 1j * np.arange(m)[:, None] / 13
    rhs_bound(bands.sin2.apply(f), s2 @ f, m)
    weighted, mean, constraint = bands.poisson_rhs(f)
    expected_mean = en @ f[:, n // 2] / en[zero]
    corrected = f.copy()
    corrected[zero, n // 2] -= expected_mean
    rhs_bound(mean, expected_mean, m)
    rhs_bound(weighted, s2 @ corrected, m)
    assert constraint == 0
    assert np.array_equal(
        np.asarray(bands.constraint_rows()),
        np.concatenate((np.arange(zero), np.arange(zero + 1, m))),
    )
    for k, c in [(0.7 + 1e-14j, 1), (1.3 + 0.4j, -0.75 + 0.2j)]:
        op, shifts, rhs, integral = bands.helmholtz(f, k, c)
        reference = c * (s2 @ d2 + cs @ d1) / k**2 + s2
        matrix_bound(op.dense(), reference, m)
        matrix_bound(shifts, c * np.diag(dn) / k**2, m)
        rhs_bound(rhs, (s2 @ f) / k**2, m)
        rhs_bound(integral, en @ f[:, n // 2] / k**2, m)
        assert np.imag(np.asarray(shifts)[0]) != 0
        matrix_bound(op.shifted(shifts[0]).dense(), reference + c * dn[0, 0] / k**2 * np.eye(m), m)


@pytest.mark.parametrize("n,count", [(5, 1), (5, 4), (5, 7), (5, 11), (3, 11)])
def test_independent_coefficient_convolution(n, count):
    raw = np.arange(1, count + 1) + 1j * np.arange(count, 0, -1) / 3
    bands = multiplier_bands(n, raw)
    # Independent centered-mode convolution, including native Nyquist split.
    if count % 2 == 0:
        coeffs = np.concatenate((raw[:1] / 2, raw[1:], raw[:1] / 2))
    else:
        coeffs = raw
    frequencies = np.arange(len(coeffs)) - len(coeffs) // 2
    a = np.zeros((n, n), dtype=complex)
    x = np.arange(1, n + 1) / 7 + 1j * np.arange(n, 0, -1) / 11
    expected = np.zeros(n, dtype=complex)
    for amplitude, frequency in zip(coeffs, frequencies):
        for source in range(n):
            target = source + frequency
            if 0 <= target < n:
                a[target, source] += amplitude
                expected[target] += amplitude * x[source]
    matrix_bound(bands.dense(), a, n)
    rhs_bound(bands.apply(x), expected, n)
    rhs_bound(
        bands.apply(np.column_stack((x, 2j * x))), np.column_stack((expected, 2j * expected)), n
    )
    assert bands.lower == min(n - 1, len(coeffs) // 2)
    assert not np.any(np.asarray(bands.ab)[:, bands.lower + bands.upper + 1 :])


@pytest.mark.parametrize("n", [7, 8])
def test_nyquist_and_dynamic_common_bandwidth(n):
    modes = np.arange(n) - n // 2
    matrix_bound(differentiation_diagonal(n, 1, nyquist=True), 1j * modes, n)
    plain = 1j * modes
    if n % 2 == 0:
        plain[0] = 0
    matrix_bound(differentiation_diagonal(n, 1), plain, n)
    cs = np.asarray([2 + 3j, 1 - 2j, -0.5 + 1j, 4 - 0.25j])
    s2 = np.asarray([1, 2j, 3, 4j, 5, 6j, 7])
    op = build_source_operators(n, n, cs_coeffs=cs, sin2_coeffs=s2)
    assert op.laplace.lower == 3
    # Injected counts differ, so common padding must not drop either support.
    from chebfunjax.operators.trigspec import multmat

    expected = np.asarray(multmat(n, s2)) @ np.diag(-(modes**2))
    expected += np.asarray(multmat(n, cs)) @ np.diag(1j * modes)
    matrix_bound(op.laplace.dense(), expected, n)


def test_reserved_fill_not_input_operator():
    bands = multiplier_bands(5, np.asarray([1.0, 2.0, 3.0]))
    original = bands.apply(np.arange(5.0))
    dirty = bands.ab.at[:, -1].set(123.0)
    assert np.array_equal(
        np.asarray(band_apply(dirty, np.arange(5.0), lower=1, upper=1)), np.asarray(original)
    )
