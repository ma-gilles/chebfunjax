"""Port of MATLAB Chebfun tests/misc/test_cheb2leg.m (Fable 5).

Provenance
----------
MATLAB source : tests/misc/test_cheb2leg.m
Chebfun commit: 7574c77
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.utils.transforms import cheb2leg, leg2cheb

TOL = 5e-12


class TestCheb2leg:
    def test_native_small_row_vector_early_return(self):
        # Literal native pass 1: [zeros(N,1);1]' has shape 1x(N+1).
        # MATLAB sees N=1 from size(c_cheb), so cheb2leg returns it unchanged.
        N = 20
        c_cheb = jnp.zeros((1, N + 1), dtype=jnp.float64).at[0, -1].set(1.0)
        c_leg = cheb2leg(c_cheb)
        assert c_cheb.shape == (1, N + 1)
        assert float(jnp.linalg.norm(c_cheb - c_leg, ord=jnp.inf)) < TOL

    def test_constant_coefficient_control(self):
        # Supplemental mathematical control in the Python low-degree-first
        # vector convention; this is distinct from the native row predicate.
        c_cheb = jnp.zeros(21, dtype=jnp.float64).at[0].set(1.0)
        c_leg = cheb2leg(c_cheb)
        assert float(jnp.max(jnp.abs(c_leg - c_cheb))) < TOL

    def test_decaying_coefficients_reference(self):
        # MATLAB: c_cheb = 1./(N:-1:1)'.^2; c_cheb(2:2:end) negated
        # (low-degree-first vector, entries (-1)^k/(N-k)^2).
        N = 20
        c = 1.0 / np.arange(N, 0, -1, dtype=float) ** 2
        c[1::2] *= -1
        c_leg = np.asarray(cheb2leg(jnp.asarray(c)))
        # MATLAB reference: c_leg(2) = 0.011460983274163
        assert abs(c_leg[1] - 0.011460983274163) / 0.011460983274163 < TOL

    def test_roundtrip(self):
        # Native source pass 3 uses the same length-20 alternating input as pass 2.
        c = 1.0 / np.arange(20, 0, -1, dtype=float) ** 2
        c[1::2] *= -1
        c = jnp.asarray(c)
        back = leg2cheb(cheb2leg(c))
        assert float(jnp.max(jnp.abs(back - c))) < TOL

    def test_native_large_row_vector_early_return(self):
        # Literal native pass 4: shape 1x1001 takes the same early return.
        N = 1000
        c_cheb = jnp.zeros((1, N + 1), dtype=jnp.float64).at[0, -1].set(1.0)
        c_leg = cheb2leg(c_cheb)
        assert c_cheb.shape == (1, N + 1)
        assert float(jnp.linalg.norm(c_cheb - c_leg, ord=jnp.inf)) < 10 * TOL

    def test_matrix_columns_match_native_vectorization(self):
        # MATLAB source pass 7: cheb2leg(A) equals independent conversion of
        # every column. Keep this small control below the native fast cutoff.
        a = jnp.asarray([[0.2, -0.1], [0.5, 0.3], [-0.4, 0.7]])
        actual = cheb2leg(a)
        separate = jnp.stack([cheb2leg(a[:, j]) for j in range(a.shape[1])], axis=1)
        assert float(jnp.linalg.norm(actual - separate)) < TOL

    def test_native_small_row_early_return_precedes_normalization(self):
        # cheb2leg.m returns c_cheb unchanged for N < 2 before either direct
        # or fast conversion (and therefore before normalize is applied).
        for c in (jnp.asarray([2.5]), jnp.asarray([[2.5, -3.0]]),
                  jnp.empty((0, 2))):
            actual = cheb2leg(c, normalize=True)
            assert actual.shape == c.shape
            np.testing.assert_array_equal(np.asarray(actual), np.asarray(c))

    def test_fast_branch_at_513_matches_direct_reference(self):
        # Independent cutoff control; the 513-row native branch is fast.
        # The direct implementation is an independent O(N^2) oracle here.
        from chebfunjax.utils.transforms import _cheb2leg_direct

        c = jnp.asarray(np.random.default_rng(11).standard_normal(513))
        fast = cheb2leg(c)
        direct = _cheb2leg_direct(c, False)
        assert float(jnp.linalg.norm(fast - direct, ord=jnp.inf)) < TOL
        compiled = jax.jit(cheb2leg)(c)
        assert float(jnp.linalg.norm(compiled - direct, ord=jnp.inf)) < TOL

    def test_fast_cold_jit_complex_matches_direct(self):
        # Cold-cache JIT is the first use of this N-specific plan; complex
        # linearity exercises the supported source dtype without a warmed eager
        # call. The direct recurrence is the same source-derived transform.
        from chebfunjax.utils.transforms import (
            _cheb2leg_cholesky_plan,
            _cheb2leg_direct,
        )

        _cheb2leg_cholesky_plan.cache_clear()
        n = 513
        real = jnp.asarray(np.random.default_rng(19).standard_normal(n))
        imag = jnp.asarray(np.random.default_rng(20).standard_normal(n))
        c = real + 1j * imag
        compiled = jax.jit(cheb2leg)(c)
        direct = _cheb2leg_direct(c, False)
        assert float(jnp.linalg.norm(compiled - direct, ord=jnp.inf)) < TOL
        linear = cheb2leg(real) + 1j * cheb2leg(imag)
        assert float(jnp.linalg.norm(compiled - linear, ord=jnp.inf)) < TOL
        _, chol = _cheb2leg_cholesky_plan(n)
        print(f"cold_jit_plan_n={n} rank={chol.shape[1]} bytes={chol.size * chol.dtype.itemsize}")

    def test_fast_plan_stops_at_source_residual(self):
        # Supplementary workspace control: cache contains only native retained
        # columns, and the residual reaches the literal source tolerance.
        from chebfunjax.utils.transforms import _cheb2leg_cholesky_plan

        n = 513
        vals, chol = _cheb2leg_cholesky_plan(n)
        num = jnp.arange(1, n, dtype=jnp.float64)
        diag0 = vals[2 * jnp.arange(1, n) - 1] * num**2 / (2 * num + 1)
        residual = diag0 - jnp.sum(chol**2, axis=1)
        assert chol.shape[1] < n - 1
        assert float(jnp.max(residual)) <= 1e-14 * np.log(n)

    def test_large_stored_coefficient(self):
        # Native source pass 5: 1000 coefficients and its stored value at
        # MATLAB index 559 (Python index 558), with the original relative bound.
        N = 1000
        c = 1.0 / np.arange(N, 0, -1, dtype=float) ** 2
        c[1::2] *= -1
        actual = np.asarray(cheb2leg(jnp.asarray(c)))[558]
        expected = -8.239505429144573e-04
        assert abs(actual - expected) / abs(expected) < 10 * TOL

    def test_large_roundtrip(self):
        # Native source pass 6; retain the original infinity-norm bound.
        N = 1000
        c = 1.0 / np.arange(N, 0, -1, dtype=float) ** 2
        c[1::2] *= -1
        actual = leg2cheb(cheb2leg(jnp.asarray(c)))
        assert float(jnp.linalg.norm(actual - jnp.asarray(c), ord=jnp.inf)) < TOL

    @pytest.mark.parametrize("N", [3, 514])
    def test_vectorization(self, N):
        # Native source passes 7/8: norm(matrix difference) uses spectral norm.
        A = jnp.asarray(np.random.default_rng(0).random((N, 10)))
        actual = cheb2leg(A)
        separate = jnp.stack([cheb2leg(A[:, jj]) for jj in range(10)], axis=1)
        assert float(jnp.linalg.norm(actual - separate, ord=2)) < TOL

    @pytest.mark.parametrize("N,scale", [(10, 10), (1000, 1)])
    def test_normalization(self, N, scale):
        # Native source passes 9/11: rowwise orthonormal scaling.
        A = jnp.asarray(np.random.default_rng(0).random((N, 2)))
        B = cheb2leg(A)
        C = cheb2leg(A, "norm")
        # Match native diag(1./sqrt(...))*B operation order.
        inv_sqrt = 1.0 / jnp.sqrt(jnp.arange(N, dtype=jnp.float64) + 0.5)
        expected = inv_sqrt[:, None] * B
        assert float(jnp.linalg.norm(expected - C, ord=2)) < scale * TOL

    @pytest.mark.parametrize("N", [10, 1000])
    def test_normalization_alias(self, N):
        # Native source passes 10/12 use 'normalized' as an equivalent option.
        A = jnp.asarray(np.random.default_rng(1).random((N, 2)))
        C = cheb2leg(A, "norm")
        D = cheb2leg(A, "normalized")
        assert float(jnp.linalg.norm(C - D, ord=2)) < TOL
