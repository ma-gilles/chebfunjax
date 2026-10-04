"""Focused ports of MATLAB @chebtech/plateauCheck.m behavior.

Provenance
----------
MATLAB source : @chebtech/plateauCheck.m
MATLAB tests : tests/chebtech/test_happinessCheck.m (pass 9)
Chebfun commit: 7574c77
"""

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.plateau import _plateau_check


def test_strict_cutoff_uses_last_coefficient_and_four_coefficient_guard():
    tol = 1e-10
    coeffs = 0.5 ** jnp.arange(100, dtype=jnp.float64)
    values = jnp.ones_like(coeffs)
    n90 = int(np.ceil(0.90*coeffs.size))
    normalized = np.asarray(coeffs[:n90])
    above = np.flatnonzero(normalized >= tol/50.0)
    expected_cutoff = 4 + int(above[-1]) + 1

    happy, cutoff = _plateau_check(coeffs, values, 1.0, tol)

    assert happy
    assert cutoff == expected_cutoff


def test_sustained_small_coefficient_plateau_is_accepted():
    n = 160
    index = jnp.arange(n, dtype=jnp.float64)
    # Independent prescribed sequence: exponential decay transitions to a
    # stable 1e-9 tail, which is below plateauCheck's 1e-7 accuracy floor.
    coeffs = jnp.where(index < 35, 0.5*jnp.exp(-0.7*index), 1e-9)

    happy, cutoff = _plateau_check(coeffs, jnp.ones(n), 1.0, 2.22e-16)

    assert happy
    assert 35 < cutoff < int(np.ceil(0.90*n))


def test_array_columns_share_happiness_and_largest_cutoff():
    n = 160
    index = jnp.arange(n, dtype=jnp.float64)
    resolved = 0.5 ** index
    plateau = jnp.where(index < 35, 0.5*jnp.exp(-0.7*index), 1e-9)
    coeffs = jnp.stack((resolved, plateau), axis=1)
    values = jnp.ones_like(coeffs)

    array_result = _plateau_check(coeffs, values, jnp.array([1.0, 1.0]),
                                  2.22e-16)
    column_results = [
        _plateau_check(coeffs[:, j], values[:, j], 1.0, 2.22e-16)
        for j in range(2)
    ]

    assert array_result[0] == all(result[0] for result in column_results)
    assert array_result[1] == max(result[1] for result in column_results)


def test_zero_and_infinite_values_follow_source_shortcuts():
    coeffs = jnp.zeros(32)
    assert _plateau_check(coeffs, coeffs, 0.0, 1e-12) == (True, 1)

    infinite_values = jnp.full(32, jnp.inf)
    assert _plateau_check(coeffs, infinite_values, 1.0, 1e-12) == (False, 32)


def test_nan_coefficients_raise_source_error():
    coeffs = jnp.ones(32).at[12].set(jnp.nan)
    with pytest.raises(ValueError, match="Function returned NaN when evaluated"):
        _plateau_check(coeffs, jnp.ones(32), 1.0, 1e-12)


@pytest.mark.parametrize("n, happy", [(5, False), (6, True), (7, True)])
def test_short_constant_preserves_source_strict_guard(n, happy):
    coeffs = jnp.zeros(n).at[0].set(1.0)
    # Source cutoff5 is below .95*ceil(.9*n) only for the last two sizes.
    assert _plateau_check(coeffs, jnp.ones(n), 1.0, 1e-12) == (happy, 5)


def test_empty_strict_cutoff_is_preserved_on_early_return():
    coeffs = jnp.zeros(8).at[0].set(1e-20)
    happy, cutoff = _plateau_check(coeffs, jnp.full(8, 1e-20), 1.0, 1e-12)
    assert not happy
    assert cutoff.size == 0


def test_empty_strict_cutoff_deletes_array_slot_before_final_max():
    # First column is tiny relative to the supplied global scale. The source
    # exits there; the second column's unvisited zero cutoff remains.
    coeffs = jnp.zeros((8, 2)).at[0].set(jnp.array([1e-20, 1.0]))
    values = jnp.broadcast_to(jnp.array([1e-20, 1.0]), (8, 2))
    assert _plateau_check(coeffs, values, 1.0, 1e-12) == (False, 0)
