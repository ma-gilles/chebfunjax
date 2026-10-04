"""Port of MATLAB Chebfun tests/chebfun/test_dlt.m.

The MATLAB test seeds its own random stream with ``seedRNG(42)``. NumPy's
``default_rng(42)`` is an explicit distribution-only adaptation, not the same
random stream. The pass-3 direct-reference check is kept separate from the
production coverage claims: its rolling recurrence is a small O(n)-memory
test oracle, while the independent pass-1/2 oracle uses NumPy's Legendre
Vandermonde constructor at the source nodes.

Provenance
----------
MATLAB source : tests/chebfun/test_dlt.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.fasttransforms import dlt
from chebfunjax.utils.quadrature import legpts

EPS = np.finfo(np.float64).eps


def _matlab_random_vector(n: int) -> np.ndarray:
    """Seeded Python analogue of MATLAB `rand(n,1)./(1:n)'`.

    This matches the source distribution and scaling, not MATLAB's RNG stream.
    """
    random = np.random.default_rng(42).random((n, 1))
    return random / np.arange(1, n + 1, dtype=np.float64)[:, None]


def _rolling_dlt_reference(coeffs: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Source `dlt_direct` recurrence, with O(n) working storage."""
    c = np.asarray(coeffs)
    vector_input = c.ndim == 1
    cm = c[:, None] if vector_input else c
    n = cm.shape[0]
    x = np.asarray(nodes, dtype=np.float64)
    if n == 0:
        out = cm.copy()
    elif n == 1:
        out = np.ones_like(cm) + 0 * cm
    else:
        pm1 = np.ones_like(x)
        p = x.copy()
        out = cm[0][None, :] + x[:, None] * cm[1][None, :]
        for k in range(1, n - 1):
            pp1 = ((2.0 - 1.0 / (k + 1.0)) * (p * x)
                   - (1.0 - 1.0 / (k + 1.0)) * pm1)
            out = out + pp1[:, None] * cm[k + 1][None, :]
            pm1, p = p, pp1
    return out[:, 0] if vector_input else out


def _small_legendre_values(nodes: np.ndarray, n: int) -> np.ndarray:
    """Independent small-N P(x) oracle, matching MATLAB `legpoly(0:n-1)`."""
    return np.polynomial.legendre.legvander(np.asarray(nodes), n - 1)


def _matrix_inf_norm(a: np.ndarray) -> float:
    """MATLAB `norm(A,inf)`: maximum absolute row sum for matrices."""
    if a.ndim == 1:
        return float(np.max(np.abs(a), initial=0.0))
    return float(np.max(np.sum(np.abs(a), axis=1), initial=0.0))


class TestChebfunDltMatlab:
    def test_scipy_legendre_polynomial_control(self):
        """Independent SciPy check retained separately from MATLAB passes."""
        from scipy.special import eval_legendre

        n = 9
        coeffs = np.zeros(n)
        coeffs[4] = 1.0
        x, _ = legpts(n)
        expected = eval_legendre(4, np.asarray(x))
        np.testing.assert_allclose(
            np.asarray(dlt(jnp.asarray(coeffs))), expected, atol=1e-13
        )

    def test_source_pass1_small_against_legendre_vandermonde(self):
        # MATLAB pass(1): norm(P(x)*r - chebfun.dlt(r), inf) < 100*n*eps.
        n = 10
        x, _ = legpts(n)
        x = np.asarray(x)
        r = _matlab_random_vector(n)[:, 0]
        expected = _small_legendre_values(x, n) @ r
        error = expected - np.asarray(dlt(jnp.asarray(r)))
        assert _matrix_inf_norm(error) < 100 * n * EPS

    def test_source_pass2_small_duplicate_columns(self):
        # MATLAB pass(2): vector input duplicated into two equal columns.
        n = 10
        x, _ = legpts(n)
        x = np.asarray(x)
        r = _matlab_random_vector(n)
        rr = np.concatenate((r, r), axis=1)
        expected = _small_legendre_values(x, n) @ rr
        error = expected - np.asarray(dlt(jnp.asarray(rr)))
        assert _matrix_inf_norm(error) < 100 * n * EPS

    def test_source_pass3_direct_oracle_matches_small_vandermonde(self):
        # Keep the original local dlt_direct-vs-P(x) assertion as a test
        # oracle check; it does not increase production-transform coverage.
        n = 10
        x, _ = legpts(n)
        x = np.asarray(x)
        r = _matlab_random_vector(n)[:, 0]
        expected = _small_legendre_values(x, n) @ r
        direct = _rolling_dlt_reference(r, x)
        assert _matrix_inf_norm(expected - direct) < 100 * n * EPS

    def test_source_pass4_large_direct_reference(self):
        # MATLAB pass(4), N=5001; preserve Tol=100*N*eps exactly.
        n = 5001
        x, _ = legpts(n)
        r = _matlab_random_vector(n)[:, 0]
        direct = _rolling_dlt_reference(r, np.asarray(x))
        result = np.asarray(dlt(jnp.asarray(r)))
        assert _matrix_inf_norm(direct - result) < 100 * n * EPS

    def test_source_pass5_large_duplicate_columns_uses_matrix_inf_norm(self):
        # MATLAB pass(5), N=5001. `norm(error,inf)` is max row sum, so two
        # equal failing columns contribute twice the scalar-column error.
        n = 5001
        x, _ = legpts(n)
        r = _matlab_random_vector(n)
        rr = np.concatenate((r, r), axis=1)
        direct = _rolling_dlt_reference(rr, np.asarray(x))
        result = np.asarray(dlt(jnp.asarray(rr)))
        assert _matrix_inf_norm(direct - result) < 100 * n * EPS

    def test_large_complex_low_mode_is_jittable(self):
        # Bounded source-branch/JIT control independent of the random oracle.
        # The exact Legendre series is evaluated analytically at source nodes.
        n = 5000
        x, _ = legpts(n)
        x = jnp.asarray(x)
        coeffs = jnp.zeros((n,), dtype=jnp.complex128)
        coeffs = coeffs.at[0].set(1.0 + 2.0j)
        coeffs = coeffs.at[1].set(-0.25 + 0.5j)
        coeffs = coeffs.at[2].set(0.125 - 0.25j)
        expected = (
            coeffs[0]
            + coeffs[1] * x
            + coeffs[2] * (3.0 * x**2 - 1.0) / 2.0
        )
        actual = jax.jit(dlt)(coeffs)
        assert actual.shape == (n,)
        assert bool(jnp.all(jnp.isfinite(actual)))
        assert float(jnp.max(jnp.abs(actual - expected))) < 100 * n * EPS

    @pytest.mark.parametrize(
        ("coeffs", "expected_shape", "expected"),
        [
            (jnp.empty((0, 2)), (0, 2), jnp.empty((0, 2))),
            (jnp.array([[2.0, -3.0]]), (1, 2), jnp.ones((1, 2))),
        ],
        ids=["empty-matrix", "single-row-matrix"],
    )
    def test_empty_and_single_row_source_contracts(
        self, coeffs, expected_shape, expected
    ):
        actual = dlt(coeffs)
        assert actual.shape == expected_shape
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
