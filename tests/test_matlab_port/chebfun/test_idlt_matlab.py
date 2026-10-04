"""Port of all seven assertions in MATLAB ``tests/chebfun/test_idlt.m``.

Provenance: Chebfun commit 7574c77680d7e82b79626300bf255498271a72df.
The MATLAB seedRNG(42) stream is not reproduced; deterministic NumPy PCG64
fixtures retain the source's ``rand(n,1)./(1:n)'`` data shape and scaling.
Source tolerances and matrix infinity norms are preserved exactly.
"""

from __future__ import annotations

import numpy as np

from chebfunjax.utils.fasttransforms import dlt, idlt
from chebfunjax.utils.quadrature import legpts


def _source_fixture(n: int, ncols: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, ncols)) / (np.arange(n, dtype=np.float64) + 1.0)[:, None]


def _legendre_vandermonde(nodes: np.ndarray, n: int) -> np.ndarray:
    """Independent P_0..P_{n-1} columns, as MATLAB ``legpoly(0:n-1)``."""
    return np.polynomial.legendre.legvander(nodes, n - 1)


def _idlt_direct_numpy(
    values: np.ndarray, nodes: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Literal test-only source recurrence with O(n * ncols) storage."""
    c = np.asarray(values, dtype=np.float64)
    vector = c.ndim == 1
    if vector:
        c = c[:, None]
    n = c.shape[0]
    weighted = np.asarray(weights, dtype=np.float64)[:, None] * c
    p_prev = np.ones_like(nodes)
    p = np.asarray(nodes, dtype=np.float64)
    out = np.zeros_like(weighted)
    out[0, :] = np.sum(weighted, axis=0)
    if n > 1:
        out[1, :] = p @ weighted
    for k in range(n - 2):
        source_n = k + 1.0
        p_next = (2.0 - 1.0 / (source_n + 1.0)) * (p * nodes) - (
            1.0 - 1.0 / (source_n + 1.0)
        ) * p_prev
        p_prev, p = p, p_next
        out[k + 2, :] = p @ weighted
    out *= (np.arange(n, dtype=np.float64) + 0.5)[:, None]
    return out[:, 0] if vector else out


def _matlab_matrix_inf_norm(values: np.ndarray) -> float:
    """MATLAB norm(A,inf): maximum absolute row sum for matrix inputs."""
    values = np.asarray(values)
    if values.ndim == 1:
        return float(np.max(np.abs(values)))
    return float(np.max(np.sum(np.abs(values), axis=1)))


class TestChebfunIdltMatlab:
    def test_source_pass1_small_basic_against_legendre_solve(self):
        n = 10
        nodes, _weights = legpts(n)
        nodes = np.asarray(nodes)
        values = _source_fixture(n, 1, seed=42)
        basis = _legendre_vandermonde(nodes, n)
        expected = np.linalg.solve(basis, values)
        actual = np.asarray(idlt(values))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(expected - actual) < tol

    def test_source_pass2_small_duplicate_columns(self):
        n = 10
        nodes, _weights = legpts(n)
        nodes = np.asarray(nodes)
        values = _source_fixture(n, 1, seed=42)
        duplicate = np.column_stack((values, values))
        basis = _legendre_vandermonde(nodes, n)
        expected = np.linalg.solve(basis, duplicate)
        actual = np.asarray(idlt(duplicate))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(expected - actual) < tol

    def test_source_pass3_small_direct_recurrence(self):
        n = 10
        nodes, weights = legpts(n)
        nodes = np.asarray(nodes)
        weights = np.asarray(weights)
        values = _source_fixture(n, 1, seed=42)
        expected = _idlt_direct_numpy(values, nodes, weights)
        actual = np.asarray(idlt(values))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(expected - actual) < tol

    def test_source_pass4_large_basic_against_direct_recurrence(self):
        n = 5001
        nodes, weights = legpts(n)
        nodes = np.asarray(nodes)
        weights = np.asarray(weights)
        values = _source_fixture(n, 1, seed=42)
        expected = _idlt_direct_numpy(values, nodes, weights)
        actual = np.asarray(idlt(values))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(expected - actual) < tol

    def test_source_pass5_large_duplicate_columns_uses_matrix_inf_norm(self):
        n = 5001
        nodes, weights = legpts(n)
        nodes = np.asarray(nodes)
        weights = np.asarray(weights)
        values = _source_fixture(n, 1, seed=42)
        duplicate = np.column_stack((values, values))
        expected = _idlt_direct_numpy(duplicate, nodes, weights)
        actual = np.asarray(idlt(duplicate))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(expected - actual) < tol

    def test_source_pass6_small_dlt_idlt_roundtrip(self):
        n = 10
        values = _source_fixture(n, 1, seed=42)
        recovered = np.asarray(dlt(idlt(values)))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(recovered - values) < tol

    def test_source_pass7_large_dlt_idlt_roundtrip(self):
        n = 5001
        values = _source_fixture(n, 1, seed=42)
        recovered = np.asarray(dlt(idlt(values)))
        tol = 100.0 * n * np.finfo(np.float64).eps
        assert _matlab_matrix_inf_norm(recovered - values) < tol

    def test_independent_roundtrip_n21(self):
        """Retain the prior non-source n=21 roundtrip control separately."""
        values = _source_fixture(21, 1, seed=2)[:, 0]
        np.testing.assert_allclose(np.asarray(idlt(dlt(values))), values, atol=1e-12)
