"""Column-wise trigonometric simplification from MATLAB source.

Provenance
----------
MATLAB source : @trigtech/simplify.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech.trigtech import (
    Trigtech,
    _trig_prolong_coeffs,
    _trig_simplify_cutoff,
    trig_coeffs2vals,
    trig_vals2coeffs,
)
from chebfunjax.utils.misc import standard_chop


def _complex_coefficients(n: int = 47) -> jnp.ndarray:
    modes = jnp.arange(n, dtype=jnp.float64) - n // 2
    amplitudes = 0.72 ** jnp.abs(modes)
    phases = jnp.where(modes % 2 == 0, 1.0j, -1.0j)
    return amplitudes.astype(jnp.complex128) * phases


def test_array_tolerances_match_separate_columns_and_shared_max_cutoff():
    coeffs = jnp.stack(
        (_complex_coefficients(), 1e-7 * _complex_coefficients()), axis=1
    )
    tolerances = jnp.asarray([1e-12, 1e-4])

    cutoff, chop_length = _trig_simplify_cutoff(coeffs, tolerances)
    independent = [
        _trig_simplify_cutoff(coeffs[:, j], tolerances[j])
        for j in range(coeffs.shape[1])
    ]
    assert cutoff == max(item[0] for item in independent)
    assert chop_length == independent[0][1] == independent[1][1]


def test_looser_small_column_tolerance_shortens_its_chop_contribution():
    n = 47
    modes = jnp.arange(n, dtype=jnp.float64) - n // 2
    low_degree = jnp.where(jnp.abs(modes) <= 2, 1.0 / (1.0 + jnp.abs(modes)), 0.0)
    slow_small_tail = 1e-7 * 0.6 ** jnp.abs(modes)
    coeffs = jnp.stack((low_degree, slow_small_tail), axis=1)

    unequal, _ = _trig_simplify_cutoff(coeffs, jnp.asarray([1e-14, 1e-2]))
    tight, _ = _trig_simplify_cutoff(coeffs, jnp.asarray([1e-14, 1e-14]))
    assert unequal < tight


@pytest.mark.parametrize("nold", [47, 48])
def test_simplify_uses_source_reversal_and_raw_pairing_before_chop(nold):
    coeffs = _complex_coefficients(nold)
    tech = Trigtech(coeffs=coeffs, is_real=False, ishappy=True)
    nold = tech.n
    padded_n = max(17, int(jnp.floor(nold * 1.25 + 5 + 0.5)))
    prolonged = tech.prolong(padded_n)

    # Literal @trigtech/simplify.m preprocessing order: abs(reverse(c)),
    # then coeffs2vals/vals2coeffs, then pair opposite Fourier modes.
    source_input = jnp.abs(prolonged.coeffs[::-1])
    noisy = trig_vals2coeffs(trig_coeffs2vals(source_input))
    # Literal MATLAB slices, with 0-based slice equivalents of its 1-based
    # indexing, followed by flipud and the duplicated-pair expansion.
    n = noisy.shape[0]
    if n % 2 == 0:
        half = n // 2
        source_rows = jnp.concatenate(
            (
                noisy[-1:],
                noisy[half:n - 1][::-1] + noisy[:half - 1],
                noisy[half - 1:half],
            )
        )[::-1]
    else:
        half = (n + 1) // 2
        source_rows = jnp.concatenate(
            (
                noisy[half:n][::-1] + noisy[:half - 1],
                noisy[half - 1:half],
            )
        )[::-1]
    chop = jnp.concatenate((source_rows[:1], jnp.repeat(source_rows[1:], 2)))
    source_cutoff = min(standard_chop(chop, 1e-13), nold)
    n_keep = source_cutoff + 1 if source_cutoff % 2 == 0 else source_cutoff
    expected = _trig_prolong_coeffs(tech.coeffs, max(1, n_keep))

    actual = tech.simplify(1e-13)
    np.testing.assert_allclose(
        np.asarray(actual.coeffs), np.asarray(expected), rtol=2e-13, atol=2e-13
    )


def test_even_length_unresolved_series_retains_split_nyquist_mode():
    # A broad, non-decaying spectrum has no chop plateau. MATLAB caps the
    # chop count at nold, then splits an even Nyquist coefficient into the
    # adjacent +/- modes when rebuilding the centered odd-length series.
    coeffs = jnp.asarray(
        [1.0, -0.4, 0.7, 0.2, -0.9, 0.3, 0.8, -0.5,
         0.6, -0.2, 0.9, 0.1, -0.8, 0.4, 0.5, -0.7],
        dtype=jnp.complex128,
    )
    tech = Trigtech(coeffs=coeffs, is_real=False, ishappy=True)
    points = jnp.linspace(-1.0, 1.0, 65, endpoint=False)

    simplified = tech.simplify(1e-16)
    assert simplified.n == tech.n + 1
    np.testing.assert_allclose(
        np.asarray(simplified(points)), np.asarray(tech(points)),
        rtol=2e-13, atol=2e-13,
    )
