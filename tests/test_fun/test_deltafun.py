"""Tests for chebfunjax.fun.deltafun — Deltafun class.

Covers distributions of the form f(x) + Σ_k c_k δ(x − x_k).

JAX contract:
- __call__: jit=YES, vmap=YES (evaluates smooth funPart only)
- sum: jit=YES
- diff: construction NOT jit-safe; result evaluation IS jit-safe
- Arithmetic: scalar mul jit=NO (construction-level)

Test cases:
- Pure smooth (no deltas): evaluation, sum, diff match Bndfun
- With single delta at 0: sum = smooth_integral + delta_mag
- With multiple deltas: merging at same location, distinct locations
- diff: shifts delta_mags by one row (delta -> delta')
- Scalar multiplication: scales funPart and all delta magnitudes
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from chebfunjax.domain import Domain
from chebfunjax.fun.bndfun import Bndfun
from chebfunjax.fun.deltafun import Deltafun, _merge_deltas

RTOL = 1e-12
ATOL = 1e-13

D_STD = Domain((-1.0, 1.0))
D_01 = Domain((0.0, 1.0))


# =============================================================================
# Tier 1: Construction
# =============================================================================


class TestDeltafunConstruction:
    """Tests for Deltafun construction.

    JAX contract: construction NOT jit-safe.
    """

    def test_from_function(self):
        """from_function creates a Deltafun with no deltas."""
        df = Deltafun.from_function(jnp.sin, D_STD)
        assert df.n_deltas == 0
        assert not df.has_deltas
        assert len(df) == df.funPart.n

    def test_from_fun(self):
        """from_fun wraps an existing Bndfun."""
        fun = Bndfun.from_function(jnp.cos, D_STD)
        df = Deltafun.from_fun(fun)
        assert df.n_deltas == 0
        assert df.funPart is fun

    def test_from_fun_and_deltas(self):
        """from_fun_and_deltas stores delta data."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        locs = jnp.array([0.0, 0.5], dtype=jnp.float64)
        mags = jnp.array([1.0, -2.0], dtype=jnp.float64)
        df = Deltafun.from_fun_and_deltas(fun, locs, mags)
        assert df.n_deltas == 2

    def test_mags_promoted_to_2d(self):
        """1-D delta_mags is promoted to shape (1, N)."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        locs = jnp.array([0.3], dtype=jnp.float64)
        mags = jnp.array([5.0], dtype=jnp.float64)
        df = Deltafun(fun, locs, mags)
        assert df.delta_mags.ndim == 2
        assert df.delta_mags.shape == (1, 1)

    def test_empty_deltas(self):
        """Deltafun with empty delta arrays."""
        fun = Bndfun.from_function(jnp.exp, D_STD)
        locs = jnp.zeros(0, dtype=jnp.float64)
        mags = jnp.zeros((1, 0), dtype=jnp.float64)
        df = Deltafun(fun, locs, mags)
        assert df.n_deltas == 0
        assert not df.has_deltas

    def test_has_deltas_true(self):
        """has_deltas is True when there are non-zero magnitudes."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64),
                      jnp.array([3.0], dtype=jnp.float64))
        assert df.has_deltas

    def test_has_deltas_false_zero_mag(self):
        """has_deltas is False when all magnitudes are zero."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64),
                      jnp.array([0.0], dtype=jnp.float64))
        assert not df.has_deltas


# =============================================================================
# Tier 2: Evaluation
# =============================================================================


class TestDeltafunEval:
    """Tests for Deltafun.__call__.

    JAX contract: __call__ jit=YES, vmap=YES.
    """

    def setup_method(self):
        fun = Bndfun.from_function(jnp.sin, D_STD)
        self.df_pure = Deltafun.from_fun(fun)
        self.df_delta = Deltafun(
            fun,
            jnp.array([0.0], dtype=jnp.float64),
            jnp.array([2.5], dtype=jnp.float64),
        )

    def test_pure_scalar(self):
        """Evaluate pure-smooth Deltafun at a scalar."""
        x = jnp.float64(0.5)
        npt.assert_allclose(float(self.df_pure(x)), float(jnp.sin(x)), rtol=1e-13)

    def test_delta_eval_ignores_delta(self):
        """Evaluation ignores delta contributions (distributional, no pointwise value)."""
        x = jnp.float64(0.5)
        # Should equal funPart(x) = sin(x), ignoring the delta at 0
        npt.assert_allclose(float(self.df_delta(x)), float(jnp.sin(x)), rtol=1e-13)

    def test_array_eval(self):
        """Evaluate at an array of points."""
        xs = jnp.linspace(-0.9, 0.9, 20, dtype=jnp.float64)
        vals = self.df_pure(xs)
        expected = jnp.sin(xs)
        npt.assert_allclose(np.array(vals), np.array(expected), rtol=1e-13)

    def test_jit(self):
        """JIT-compiled evaluation via lambda wrapper."""
        df = self.df_pure
        jit_f = jax.jit(lambda x: df(x))
        x = jnp.float64(0.3)
        npt.assert_allclose(float(jit_f(x)), float(df(x)), rtol=1e-15)

    def test_vmap(self):
        """vmap evaluation."""
        df = self.df_pure
        xs = jnp.linspace(-0.8, 0.8, 10, dtype=jnp.float64)
        vmap_vals = jax.vmap(df)(xs)
        direct_vals = df(xs)
        npt.assert_allclose(np.array(vmap_vals), np.array(direct_vals), rtol=1e-15)


# =============================================================================
# Tier 3: Definite integral (sum)
# =============================================================================


class TestDeltafunSum:
    """Tests for Deltafun.sum.

    JAX contract: sum jit=YES.
    """

    def test_no_deltas_matches_bndfun(self):
        """sum with no deltas equals Bndfun.sum."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        df = Deltafun.from_fun(fun)
        npt.assert_allclose(float(df.sum()), float(fun.sum()), rtol=1e-15)

    def test_single_delta_adds_magnitude(self):
        """sum = smooth_integral + delta_magnitude."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64),
                      jnp.array([2.5], dtype=jnp.float64))
        expected = float(fun.sum()) + 2.5
        npt.assert_allclose(float(df.sum()), expected, rtol=1e-14)

    def test_multiple_deltas(self):
        """sum = smooth_integral + sum of all delta magnitudes."""
        fun = Bndfun.from_function(jnp.cos, D_STD)
        locs = jnp.array([-0.5, 0.0, 0.5], dtype=jnp.float64)
        mags = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float64)
        df = Deltafun(fun, locs, mags)
        expected = float(fun.sum()) + (1.0 - 2.0 + 3.0)
        npt.assert_allclose(float(df.sum()), expected, rtol=1e-13)

    def test_higher_order_delta_not_counted(self):
        """Higher-order delta derivatives contribute 0 to the integral."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        # Row 0: 0 (no plain delta), Row 1: 3.0 (delta prime at x=0)
        mags_2d = jnp.array([[0.0], [3.0]], dtype=jnp.float64)
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64), mags_2d)
        # Only row-0 counts for integration
        expected = float(fun.sum())
        npt.assert_allclose(float(df.sum()), expected, rtol=1e-14)

    def test_sum_jit(self):
        """sum is JIT-safe via lambda."""
        fun = Bndfun.from_function(jnp.sin, D_STD)
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64),
                      jnp.array([2.5], dtype=jnp.float64))
        jit_sum = jax.jit(lambda: df.sum())
        npt.assert_allclose(float(jit_sum()), float(df.sum()), rtol=1e-15)

    def test_negative_delta(self):
        """Negative delta magnitude reduces the sum."""
        fun = Bndfun.from_function(lambda x: jnp.ones_like(x, dtype=jnp.float64), D_STD)
        # integral of 1 on [-1,1] = 2
        df = Deltafun(fun, jnp.array([0.0], dtype=jnp.float64),
                      jnp.array([-1.0], dtype=jnp.float64))
        npt.assert_allclose(float(df.sum()), 2.0 - 1.0, rtol=1e-14)


# =============================================================================
# Tier 4: Differentiation
# =============================================================================


class TestDeltafunDiff:
    """Tests for Deltafun.diff.

    JAX contract: construction NOT jit-safe; result evaluation IS jit-safe.
    """

    def setup_method(self):
        fun = Bndfun.from_function(jnp.sin, D_STD)
        self.df = Deltafun(
            fun,
            jnp.array([0.0], dtype=jnp.float64),
            jnp.array([2.5], dtype=jnp.float64),
        )

    def test_diff_shifts_mags(self):
        """diff prepends a zero row, shifting delta to delta'."""
        df2 = self.df.diff()
        assert df2.delta_mags.shape == (2, 1)
        assert abs(float(df2.delta_mags[0, 0])) < 1e-15  # new row 0 = 0
        npt.assert_allclose(float(df2.delta_mags[1, 0]), 2.5, rtol=1e-15)

    def test_diff_funpart(self):
        """diff differentiates the funPart (sin -> cos)."""
        df2 = self.df.diff()
        x = jnp.float64(0.5)
        npt.assert_allclose(float(df2(x)), float(jnp.cos(x)), rtol=1e-12)

    def test_diff_k0(self):
        """diff(k=0) returns a copy."""
        df0 = self.df.diff(k=0)
        assert df0.delta_mags.shape == self.df.delta_mags.shape
        x = jnp.float64(0.3)
        npt.assert_allclose(float(df0(x)), float(self.df(x)), rtol=1e-15)

    def test_diff_k2(self):
        """diff(k=2) prepends 2 zero rows."""
        df2 = self.df.diff(k=2)
        assert df2.delta_mags.shape == (3, 1)
        assert abs(float(df2.delta_mags[0, 0])) < 1e-15
        assert abs(float(df2.delta_mags[1, 0])) < 1e-15
        npt.assert_allclose(float(df2.delta_mags[2, 0]), 2.5, rtol=1e-15)

    def test_diff_no_deltas(self):
        """diff on pure-smooth Deltafun produces no deltas."""
        fun = Bndfun.from_function(jnp.cos, D_STD)
        df = Deltafun.from_fun(fun)
        df2 = df.diff()
        assert df2.n_deltas == 0
        x = jnp.float64(0.4)
        npt.assert_allclose(float(df2(x)), float(-jnp.sin(x)), rtol=1e-12)

    def test_diff_result_jit(self):
        """Result of diff is JIT-safe for evaluation via lambda wrapper."""
        df2 = self.df.diff()
        jit_eval = jax.jit(lambda x: df2(x))
        x = jnp.float64(0.5)
        npt.assert_allclose(float(jit_eval(x)), float(df2(x)), rtol=1e-15)


# =============================================================================
# Tier 5: Arithmetic
# =============================================================================


class TestDeltafunArithmetic:
    """Tests for Deltafun arithmetic."""

    def setup_method(self):
        fun1 = Bndfun.from_function(jnp.sin, D_STD)
        fun2 = Bndfun.from_function(jnp.cos, D_STD)
        self.df1 = Deltafun(
            fun1,
            jnp.array([0.0], dtype=jnp.float64),
            jnp.array([2.5], dtype=jnp.float64),
        )
        self.df2 = Deltafun(
            fun2,
            jnp.array([0.5], dtype=jnp.float64),
            jnp.array([1.0], dtype=jnp.float64),
        )

    def test_add_deltafuns(self):
        """Adding two Deltafuns merges both parts."""
        df3 = self.df1 + self.df2
        assert df3.n_deltas == 2
        expected = float(self.df1.sum()) + float(self.df2.sum())
        npt.assert_allclose(float(df3.sum()), expected, rtol=1e-13)

    def test_add_coincident_deltas(self):
        """Adding two Deltafuns with same delta location merges magnitudes."""
        fun1 = Bndfun.from_function(jnp.sin, D_STD)
        fun2 = Bndfun.from_function(jnp.cos, D_STD)
        df_a = Deltafun(fun1, jnp.array([0.0], dtype=jnp.float64),
                        jnp.array([1.0], dtype=jnp.float64))
        df_b = Deltafun(fun2, jnp.array([0.0], dtype=jnp.float64),
                        jnp.array([2.0], dtype=jnp.float64))
        df3 = df_a + df_b
        assert df3.n_deltas == 1
        npt.assert_allclose(float(df3.delta_mags[0, 0]), 3.0, rtol=1e-14)

    def test_neg(self):
        """Unary negation negates funPart and delta_mags."""
        df3 = -self.df1
        npt.assert_allclose(float(df3.sum()), -float(self.df1.sum()), rtol=1e-14)
        npt.assert_allclose(
            float(df3.delta_mags[0, 0]), -2.5, rtol=1e-15
        )

    def test_sub(self):
        """Subtraction."""
        df3 = self.df1 - self.df1
        npt.assert_allclose(float(df3.sum()), 0.0, atol=1e-13)

    def test_scalar_mul(self):
        """Scalar multiplication scales both funPart and delta_mags."""
        df3 = self.df1 * 3.0
        npt.assert_allclose(float(df3.sum()), 3.0 * float(self.df1.sum()), rtol=1e-13)
        npt.assert_allclose(float(df3.delta_mags[0, 0]), 7.5, rtol=1e-15)

    def test_rmul(self):
        """c * Deltafun."""
        df3 = 2.0 * self.df1
        npt.assert_allclose(float(df3.sum()), 2.0 * float(self.df1.sum()), rtol=1e-13)

    def test_truediv(self):
        """f / scalar."""
        df3 = self.df1 / 2.0
        npt.assert_allclose(float(df3.sum()), float(self.df1.sum()) / 2.0, rtol=1e-13)

    def test_radd_bndfun(self):
        """Bndfun + Deltafun."""
        fun = Bndfun.from_function(jnp.exp, D_STD)
        df3 = self.df1 + fun
        expected = float(self.df1.sum()) + float(fun.sum())
        npt.assert_allclose(float(df3.sum()), expected, rtol=1e-13)

    def test_radd_scalar(self):
        """scalar + Deltafun."""
        df3 = self.df1 + 1.0
        # integral of 1 on [-1,1] = 2
        expected = float(self.df1.sum()) + 2.0
        npt.assert_allclose(float(df3.sum()), expected, rtol=1e-12)


# =============================================================================
# Tier 6: Merge helpers
# =============================================================================


class TestMergeDeltas:
    """Tests for the _merge_deltas helper."""

    def test_disjoint_locs(self):
        """Merging non-overlapping locations returns all of them."""
        locs1 = jnp.array([0.0, 0.5], dtype=jnp.float64)
        mags1 = jnp.array([[1.0, 2.0]], dtype=jnp.float64)
        locs2 = jnp.array([-0.5], dtype=jnp.float64)
        mags2 = jnp.array([[3.0]], dtype=jnp.float64)
        new_locs, new_mags = _merge_deltas(locs1, mags1, locs2, mags2)
        assert len(new_locs) == 3
        # Sorted: -0.5, 0.0, 0.5
        npt.assert_allclose(np.array(new_locs), [-0.5, 0.0, 0.5], rtol=1e-15)

    def test_coincident_locs_summed(self):
        """Coincident locations have their magnitudes summed."""
        locs1 = jnp.array([0.0], dtype=jnp.float64)
        mags1 = jnp.array([[1.5]], dtype=jnp.float64)
        locs2 = jnp.array([0.0], dtype=jnp.float64)
        mags2 = jnp.array([[2.5]], dtype=jnp.float64)
        new_locs, new_mags = _merge_deltas(locs1, mags1, locs2, mags2)
        assert len(new_locs) == 1
        npt.assert_allclose(float(new_mags[0, 0]), 4.0, rtol=1e-15)

    def test_empty_first(self):
        """Merging empty first list returns second list."""
        locs1 = jnp.zeros(0, dtype=jnp.float64)
        mags1 = jnp.zeros((1, 0), dtype=jnp.float64)
        locs2 = jnp.array([0.3], dtype=jnp.float64)
        mags2 = jnp.array([[7.0]], dtype=jnp.float64)
        new_locs, new_mags = _merge_deltas(locs1, mags1, locs2, mags2)
        assert len(new_locs) == 1
        npt.assert_allclose(float(new_mags[0, 0]), 7.0, rtol=1e-15)

    def test_empty_both(self):
        """Merging two empty lists returns empty."""
        locs1 = jnp.zeros(0, dtype=jnp.float64)
        mags1 = jnp.zeros((1, 0), dtype=jnp.float64)
        locs2 = jnp.zeros(0, dtype=jnp.float64)
        mags2 = jnp.zeros((1, 0), dtype=jnp.float64)
        new_locs, new_mags = _merge_deltas(locs1, mags1, locs2, mags2)
        assert len(new_locs) == 0
