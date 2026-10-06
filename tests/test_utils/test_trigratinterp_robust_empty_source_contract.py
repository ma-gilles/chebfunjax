"""Focused unrun controls for robust retries and empty source degrees/roots.

Provenance
----------
MATLAB Chebfun commit: 7574c77
Sources: trigratinterp.m (trig_rat_interp, chopCoeffs, degrees),
    @trigtech/roots.m (empty early return).
These controls supplement, and do not change bounds in, source test fixtures.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from chebfunjax.utils import ratapprox as candidate


def test_source_empty_chop_and_half_degree_convention():
    # Source chopCoeffs returns its input unchanged for empty and length <=1.
    empty = jnp.empty((0,), dtype=jnp.complex128)
    np.testing.assert_array_equal(candidate._trigrat_chop_coeffs(empty, 1.0), empty)
    np.testing.assert_array_equal(candidate._trigrat_chop_coeffs(
        jnp.asarray([3.0 + 0j]), 1.0), jnp.asarray([3.0 + 0j]))
    # MATLAB computes (length-1)/2 literally, including the empty case.
    assert candidate._trigrat_degree(empty) == -0.5
    assert candidate._trigrat_degree(jnp.asarray([1.0 + 0j])) == 0.0
    assert candidate._trigrat_degree(jnp.asarray([1.0, 0.0, 1.0])) == 1.0


def test_source_roots_returns_empty_for_empty_trigtech_without_piece_access():
    # Pinned @trigtech/roots.m tests isempty(f) before indexing columns/pieces.
    roots = candidate._trigrat_source_x_roots(
        jnp.empty((0,), dtype=jnp.complex128), -1.0, 1.0)
    assert roots.shape == (0,)
    assert jnp.issubdtype(roots.dtype, jnp.complexfloating)


def test_robust_second_iteration_slices_at_actual_matrix_width(monkeypatch):
    calls = []

    def fake_svd(matrix, full_matrices=True):
        rows, cols = matrix.shape
        calls.append((rows, cols))
        singular_count = min(rows, cols)
        if len(calls) == 1:
            # Rank profile requests a single denominator-degree reduction.
            singular = jnp.concatenate((jnp.ones((singular_count - 2,)),
                                        jnp.zeros((2,))))
        else:
            singular = jnp.ones((singular_count,))
        # Last right singular vector is exactly the constant numerator basis
        # vector. Its chopped numerator has length 1, so next source m is 0.0.
        permutation = list(range(cols))
        permutation[0], permutation[-1] = permutation[-1], permutation[0]
        V = jnp.eye(cols, dtype=jnp.float64)[:, jnp.asarray(permutation)]
        vh = V.T
        return jnp.eye(rows), singular, vh

    monkeypatch.setattr(candidate.jnp.linalg, "svd", fake_svd)
    ac, bc = candidate._trigrat_fit_coefficients(
        fk=jnp.ones((9,)), m=1, n=2,
        th=jnp.linspace(-1.0, 1.0, 9), f_even=False, f_odd=False,
        robustness=True, interpolation=False, threshold=0.1,
    )
    assert calls == [(9, 8), (9, 4)]
    assert ac.shape == (1,)
    assert bc.size == 0
