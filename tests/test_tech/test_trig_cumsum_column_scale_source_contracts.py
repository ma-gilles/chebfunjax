"""Column-wise mean guard from MATLAB @trigtech/cumsum.m."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import Trigtech, _trig_cumsum_coeffs

EPS = np.finfo(np.float64).eps


def _tech(coeffs: jnp.ndarray) -> Trigtech:
    return Trigtech(coeffs=coeffs, is_real=True)


def _matlab_continuous_cumsum_coeffs(coeffs: np.ndarray, order: int) -> np.ndarray:
    """Independent coefficient oracle transcribed from cumsumContinuousDim."""
    source = np.array(coeffs, dtype=np.complex128, copy=True)
    n = source.shape[0]
    even = n % 2 == 0
    if even:
        source[0] *= 0.5
        source = np.concatenate([source, source[:1]], axis=0)
        highest = n // 2
    else:
        highest = (n - 1) // 2
    indices = np.arange(-highest, highest + 1, dtype=np.float64)
    factors = np.zeros_like(indices, dtype=np.complex128)
    nonzero = indices != 0
    factors[nonzero] = (-1j / indices[nonzero] / np.pi) ** order
    source[highest] = 0.0
    result = source * factors.reshape((len(factors),) + (1,) * (source.ndim - 1))
    if even and order % 2 == 1:
        result[0] = 0.0
        result[n] = 0.0
    signs = (-1.0) ** indices
    result[highest] = -np.sum(result * signs.reshape(
        (len(signs),) + (1,) * (result.ndim - 1)), axis=0)
    return result[:-1] if even else result


def test_cumsum_checks_each_column_against_its_own_vscale() -> None:
    # The first column sets a very large aggregate scale. The second column's
    # mean is tiny in absolute terms but large relative to that column; source
    # @trigtech/cumsum.m:89-93 must reject it independently.
    coeffs = jnp.zeros((5, 2), dtype=jnp.complex128)
    coeffs = coeffs.at[0, 0].set(1.0e12)
    coeffs = coeffs.at[4, 0].set(1.0e12)
    coeffs = coeffs.at[1, 1].set(0.5)
    coeffs = coeffs.at[3, 1].set(0.5)
    coeffs = coeffs.at[2, 1].set(1.0e-8)

    with pytest.raises(ValueError) as exc:
        _tech(coeffs).cumsum()
    assert "CHEBFUN:TRIGTECH:cumsum:meanNotZero" in str(exc.value)
    assert (
        "Indefinite integrals are only possible for TRIGTECH objects "
        "with zero mean."
    ) in str(exc.value)


def test_cumsum_accepts_per_column_roundoff_mean_and_integrates_matrix() -> None:
    # A column's mean within its own 10*eps*vscale allowance is accepted;
    # the second column is exactly zero-mean. Odd coefficient count exercises
    # the ordinary symmetric Fourier layout.
    coeffs = jnp.zeros((5, 2), dtype=jnp.complex128)
    coeffs = coeffs.at[1, 0].set(0.5)
    coeffs = coeffs.at[3, 0].set(0.5)
    coeffs = coeffs.at[2, 0].set(5.0 * EPS)
    coeffs = coeffs.at[1, 1].set(-0.25j)
    coeffs = coeffs.at[3, 1].set(0.25j)

    result = _tech(coeffs).cumsum()
    points = jnp.array([-0.75, -0.1, 0.4])
    # With c(-1)=-i/4 and c(+1)=i/4, the stored series is
    # -0.5*sin(pi*x), whose primitive vanishing at -1 is
    # 0.5*(cos(pi*x)+1)/pi.
    expected = jnp.stack(
        [jnp.sin(jnp.pi * points) / jnp.pi,
         0.5 * (jnp.cos(jnp.pi * points) + 1.0) / jnp.pi],
        axis=1,
    )
    np.testing.assert_allclose(np.asarray(result(points)), np.asarray(expected),
                               rtol=100 * EPS, atol=100 * EPS)


def test_cumsum_zero_mean_even_coefficient_matrix() -> None:
    # The even layout has a Nyquist slot at the first row. Ordinary k=1 modes
    # integrate while retaining the source's even-layout handling.
    coeffs = jnp.zeros((4, 2), dtype=jnp.complex128)
    coeffs = coeffs.at[1, 0].set(0.5)
    coeffs = coeffs.at[3, 0].set(0.5)
    coeffs = coeffs.at[1, 1].set(-0.25j)
    coeffs = coeffs.at[3, 1].set(0.25j)

    result = _tech(coeffs).cumsum()
    points = jnp.array([-0.75, -0.1, 0.4])
    # The conjugate pair c(-1)=-i/4, c(+1)=i/4 represents
    # -0.5*sin(pi*x); integrate it and set the value at -1 to zero.
    expected = jnp.stack(
        [jnp.sin(jnp.pi * points) / jnp.pi,
         0.5 * (jnp.cos(jnp.pi * points) + 1.0) / jnp.pi],
        axis=1,
    )
    np.testing.assert_allclose(np.asarray(result(points)), np.asarray(expected),
                               rtol=100 * EPS, atol=100 * EPS)


def test_cumsum_rejects_tiny_nonzero_constant_mean() -> None:
    # For a constant column, source vscale equals |c0|; its relative guard
    # rejects every nonzero constant mean, even when its absolute magnitude
    # is far below the former unit fallback threshold.
    coeffs = jnp.array([0.0, 1.0e-20, 0.0], dtype=jnp.complex128)
    with pytest.raises(ValueError, match="CHEBFUN:TRIGTECH:cumsum:meanNotZero"):
        _tech(coeffs).cumsum()


@pytest.mark.parametrize("n", [5, 6])
@pytest.mark.parametrize("order", [1, 2])
def test_continuous_cumsum_coefficients_match_source_recurrence(
    n: int, order: int,
) -> None:
    # Nonzero modes include an even-layout Nyquist coefficient. The source
    # zeros that endpoint for odd order but retains its even-order contribution.
    coeffs = np.zeros((n, 2), dtype=np.complex128)
    coeffs[n // 2 - 1] = [0.25, -0.125j]
    coeffs[n // 2 + 1] = [0.25, 0.125j]
    if n % 2 == 0:
        coeffs[0] = [0.2, 0.1j]
    expected = _matlab_continuous_cumsum_coeffs(coeffs, order)
    actual = np.asarray(_trig_cumsum_coeffs(jnp.asarray(coeffs), m=order))
    np.testing.assert_allclose(actual, expected, rtol=100 * EPS, atol=100 * EPS)


def test_cumsum_empty_none_and_zero_order_contracts() -> None:
    empty = _tech(jnp.zeros((0,), dtype=jnp.complex128))
    assert empty.cumsum() is empty

    coeffs = jnp.array([0.5, 0.0, 0.5], dtype=jnp.complex128)
    f = _tech(coeffs)
    assert f.cumsum(m=0) is f
    np.testing.assert_allclose(
        np.asarray(f.cumsum(m=None).coeffs),
        np.asarray(f.cumsum(m=1).coeffs),
        rtol=0.0,
        atol=0.0,
    )


def test_cumsum_second_order_nonnyquist_analytic_primitive() -> None:
    # f(x)=cos(pi*x); two source integrations with F(-1)=0 give
    # F(x)=-(cos(pi*x)+1)/pi**2.
    points = jnp.array([-0.75, -0.1, 0.4])
    f = _tech(jnp.array([0.5, 0.0, 0.5], dtype=jnp.complex128))
    expected = -(jnp.cos(jnp.pi * points) + 1.0) / jnp.pi**2
    np.testing.assert_allclose(
        np.asarray(f.cumsum(m=2)(points)), np.asarray(expected),
        rtol=100 * EPS, atol=100 * EPS,
    )


def test_cumsum_finite_dimension_repeats_column_prefix_sum() -> None:
    coeffs = jnp.arange(12, dtype=jnp.float64).reshape(4, 3).astype(jnp.complex128)
    f = _tech(coeffs)
    expected = jnp.cumsum(jnp.cumsum(coeffs, axis=1), axis=1)
    actual = f.cumsum(m=2, dim=2)
    np.testing.assert_array_equal(np.asarray(actual.coeffs), np.asarray(expected))


# Provenance: MATLAB @trigtech/cumsum.m, source commit 7574c77. Its continuous-
# dimension guard is any(abs(c(ind,:)) > 1e1*vscale(f)*eps), with vscale
# evaluated per column for array-valued techs.
