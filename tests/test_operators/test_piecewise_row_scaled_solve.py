"""MATLAB row equilibration for the piecewise linear collocation solve.

Provenance
----------
MATLAB source : @valsDiscretization/mldivide.m
Chebfun commit: 7574c77
Original authors: Copyright 2017 by The University of Oxford and
    The Chebfun Developers.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest

import chebfunjax.operators.chebop as chebop_module
from chebfunjax.operators.chebop import (
    Chebop,
    _piecewise_row_scaled_solve,
    _piecewise_row_scaled_system,
)

_EPS = np.finfo(np.float64).eps


def _assert_source_solution(matrix, solution, rhs, expected):
    expected = np.asarray(expected)
    solution = np.asarray(solution)
    bound = 100 * _EPS * np.maximum(1.0, np.abs(expected))
    assert np.all(np.abs(solution - expected) <= bound)

    residual = np.abs(np.asarray(matrix) @ solution - np.asarray(rhs))
    denominator = (np.abs(np.asarray(matrix)) @ np.abs(solution)
                   + np.abs(np.asarray(rhs)))
    backward_error = np.divide(
        residual, denominator,
        out=np.zeros_like(residual, dtype=np.float64),
        where=denominator != 0,
    )
    assert np.all(backward_error <= 100 * _EPS)


def test_source_floor_one_scaling_for_subunit_and_zero_rows(monkeypatch):
    """Rows below one and zero rows keep scale one; RHS rows follow A."""
    matrix = jnp.asarray([
        [0.25, -0.5, 0.0],
        [0.0, 0.0, 0.0],
        [2.0, -4.0, 2.0],
    ])
    rhs = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scaled_a, scaled_b = _piecewise_row_scaled_system(matrix, rhs)
    expected_a = np.asarray(matrix).copy()
    expected_a[2] /= 4.0
    expected_b = np.asarray(rhs).copy()
    expected_b[2] /= 4.0
    npt.assert_array_equal(np.asarray(scaled_a), expected_a)
    npt.assert_array_equal(np.asarray(scaled_b), expected_b)

    # Observe the actual operands at the general solve boundary as well.
    solve_matrix = jnp.asarray([
        [0.25, -0.5, 0.0],
        [0.0, 0.125, 0.0],
        [2.0, -4.0, 2.0],
    ])
    solve_rhs = jnp.asarray([[1.0], [2.0], [3.0]])
    captured = {}
    original_solve = jnp.linalg.solve

    def recording_solve(a, b):
        captured["a"] = np.asarray(a)
        captured["b"] = np.asarray(b)
        return original_solve(a, b)

    monkeypatch.setattr(jnp.linalg, "solve", recording_solve)
    _piecewise_row_scaled_solve(solve_matrix, solve_rhs)
    expected_solve_a = np.asarray(solve_matrix).copy()
    expected_solve_a[2] /= 4.0
    expected_solve_b = np.asarray(solve_rhs).copy()
    expected_solve_b[2] /= 4.0
    npt.assert_array_equal(captured["a"], expected_solve_a)
    npt.assert_array_equal(captured["b"], expected_solve_b)


@pytest.mark.parametrize("tiny_row_factor", [1.0, 1.0e-12])
def test_real_badly_scaled_vector_rhs_matches_known_solution(
    tiny_row_factor,
):
    matrix = np.asarray([
        [2.0 * tiny_row_factor, 1.0 * tiny_row_factor,
         -1.0 * tiny_row_factor],
        [3.0e12, -1.0e12, 2.0e12],
        [1.0, 2.0, 4.0],
    ])
    expected = np.asarray([1.25, -0.75, 2.0])
    rhs = matrix @ expected
    got = _piecewise_row_scaled_solve(matrix, rhs)
    _assert_source_solution(matrix, got, rhs, expected)


def test_complex_matrix_multiple_rhs_matches_known_solution():
    matrix = np.asarray([
        [1.0 + 2.0j, 2.0 - 1.0j],
        [1.0e10 - 2.0e10j, -3.0e10 + 1.0e10j],
    ])
    expected = np.asarray([
        [1.0 + 0.5j, -2.0j],
        [2.0 - 1.0j, 0.25 + 1.5j],
    ])
    rhs = matrix @ expected
    got = _piecewise_row_scaled_solve(matrix, rhs)
    _assert_source_solution(matrix, got, rhs, expected)


def test_real_matrix_complex_rhs_is_not_cast_to_real():
    matrix = np.asarray([[2.0, -1.0], [3.0, 4.0]])
    expected = np.asarray([1.0 + 2.0j, -0.5 + 0.25j])
    rhs = matrix @ expected
    got = _piecewise_row_scaled_solve(matrix, rhs)
    assert np.iscomplexobj(np.asarray(got))
    _assert_source_solution(matrix, got, rhs, expected)


def test_actual_piecewise_linear_solve_uses_jax_helper(monkeypatch):
    calls = []
    source_solve = chebop_module._piecewise_row_scaled_solve

    def recording_solve(matrix, rhs):
        calls.append((np.shape(matrix), np.shape(rhs)))
        return source_solve(matrix, rhs)

    monkeypatch.setattr(
        chebop_module, "_piecewise_row_scaled_solve", recording_solve)
    problem = Chebop(lambda x, u: u.diff(2), (-1.0, 0.0, 1.0), -1.0, 1.0)
    solution = problem.solve(0.0)

    assert calls
    assert problem._pw_linear_used
    points = jnp.asarray([-1.0, -0.4, 0.0, 0.6, 1.0])
    npt.assert_allclose(
        np.asarray(solution(points)), np.asarray(points), rtol=0, atol=1e-10)


def test_nonfinite_jax_linear_result_uses_existing_fallback(monkeypatch):
    calls = []

    def nonfinite_solve(matrix, rhs):
        calls.append((np.shape(matrix), np.shape(rhs)))
        return jnp.full(jnp.shape(rhs), jnp.nan, dtype=jnp.float64)

    monkeypatch.setattr(
        chebop_module, "_piecewise_row_scaled_solve", nonfinite_solve)
    problem = Chebop(lambda x, u: u.diff(2), (-1.0, 0.0, 1.0), -1.0, 1.0)
    solution = problem.solve(0.0)

    assert calls
    assert not problem._pw_linear_used
    points = jnp.asarray([-1.0, -0.4, 0.0, 0.6, 1.0])
    npt.assert_allclose(
        np.asarray(solution(points)), np.asarray(points), rtol=0, atol=1e-10)
