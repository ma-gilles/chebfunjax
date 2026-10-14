"""Source predicates for MATLAB tests/spherefunv/test_times_divide.m.

The source comparison uses Spherefunv's global Frobenius norm, not a sampled
componentwise maximum.  Python expressions below call named source operations
where MATLAB's distinct ``.*``, ``./``, ``.\\`` and ``\\`` syntax has no
distinct Python spelling.

Provenance
----------
MATLAB source : tests/spherefunv/test_times_divide.m
Chebfun commit: 7574c77
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import Chebfun2
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.spherefun.spherefun import Spherefun
from chebfunjax.spherefun.spherefunv import Spherefunv

jax.config.update("jax_enable_x64", True)


def _sph(fc):
    def f(lam, th):
        x = jnp.cos(lam) * jnp.sin(th)
        y = jnp.sin(lam) * jnp.sin(th)
        z = jnp.cos(th)
        return fc(x, y, z)
    return Spherefun.from_function(f)


def _sampled_vnorm(F, n=25):
    lam = jnp.linspace(-np.pi + 1e-6, np.pi - 1e-6, n)
    th = jnp.linspace(1e-3, np.pi - 1e-3, n)
    L, T = jnp.meshgrid(lam, th)
    return max(float(jnp.max(jnp.abs(jnp.asarray(c(L, T)))))
               for c in F.components)


class TestSpherefunvTimesDivide:
    def test_supplemental_native_empty_short_circuits(self):
        empty = Spherefunv.empty()
        assert empty.mrdivide(2).isempty()
        assert Spherefunv.rdivide(empty, 2).isempty()
        assert Spherefunv.ldivide(2, empty).isempty()
        assert Spherefunv.mldivide(2, empty).isempty()

    def test_all_ten_original_matlab_predicates(self):
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        f = _sph(lambda x, y, z: jnp.sin(y * z) + jnp.cos(2 * x * y))
        F = Spherefunv(f, f, f)  # native [f; f; f]
        G = Spherefunv(2 * f, 2 * f, 2 * f)
        H = Spherefunv(f / 2, f / 2, f / 2)
        K = Spherefunv(f ** 2, f ** 2, f ** 2)

        # Native pass(1): norm(2*F - G) < tol.
        assert float((2 * F - G).norm()) < tol
        # Native pass(2): norm(2.*F - G) < tol.
        assert float((Spherefunv.times(2, F) - G).norm()) < tol
        # Native pass(3): norm(F*2 - G) < tol.
        assert float((F * 2 - G).norm()) < tol
        # Native pass(4): norm(F.*2 - G) < tol.
        assert float((F.times(2) - G).norm()) < tol
        # Native pass(5): norm(F/2 - H) < tol.
        assert float((F / 2 - H).norm()) < tol
        # Native pass(6): norm(F./2 - H) < tol.
        assert float((F.rdivide(2) - H).norm()) < tol
        assert float((Spherefunv.rdivide(F, 2) - H).norm()) < tol
        # Native pass(7): norm(2.\\F - H) < tol.
        assert float((Spherefunv.ldivide(2, F) - H).norm()) < tol
        # Native pass(8): norm(2\\F - H) < tol.
        assert float((Spherefunv.mldivide(2, F) - H).norm()) < tol
        # Native pass(9): norm(F.^2 - K) < tol.
        assert float((F.power(2) - K).norm()) < tol
        # Native pass(10): norm(F.*F - K) < tol.
        assert float((F.times(F) - K).norm()) < tol

        # Source error branches: mrdivide accepts only scalar-valued
        # denominators, mldivide requires a scalar left operand, and rdivide
        # rejects unsupported argument types.
        with pytest.raises(ValueError, match="mrdivide:nonScalar"):
            F.mrdivide(F)
        # Native mrdivide does not accept Spherefun (only double/Chebfun2).
        with pytest.raises(ValueError, match="mrdivide:nonScalar"):
            F.mrdivide(f)
        # Pinned MATLAB accepts this class at dispatch, then the component
        # separable-approximation divide reports its domain mismatch.
        denominator = Chebfun2.from_function(
            lambda x, y: 2 + 0 * x, domain=(-np.pi, np.pi, 0.0, np.pi))
        with pytest.raises(ValueError, match="rdivide:domains: Domains inconsistent"):
            F / denominator
        compatible_denominator = Chebfun2.from_function(
            lambda x, y: 2 + 0 * x,
            domain=(-np.pi, np.pi, -np.pi, np.pi))
        assert float(((F / compatible_denominator) - H).norm()) < tol
        # MATLAB accepts any double here, then the separableApprox scalar
        # operation reports its native size error for a nonscalar double.
        with pytest.raises(ValueError, match="MATLAB:dimagree"):
            Spherefunv.mldivide([2, 2], F)
        with pytest.raises(ValueError, match="rdivide:inputs"):
            Spherefunv.rdivide(F, object())
        # Native mtimes rejects Spherefun as a right operand for Spherefunv.
        with pytest.raises(ValueError, match="mtimes:inputs"):
            F.mtimes(f)
        # Native numeric-left 1-by-3 mtimes contracts to a scalar Spherefun.
        row_dot = Spherefunv.mtimes(jnp.asarray([[1.0, 2.0, 3.0]]), F)
        assert isinstance(row_dot, Spherefun)
        assert float((row_dot - 6 * f).norm()) < tol
        # Reflected numeric matrix dispatch preserves MATLAB's left operand.
        reflected = jnp.asarray([[1.0, 2.0, 3.0]]) * F
        assert isinstance(reflected, Spherefun)
        assert float((reflected - row_dot).norm()) < tol
        row = F.transpose()
        left_scaled_row = 2 * row
        right_scaled_row = row * 2
        assert getattr(left_scaled_row, "_row", False)
        assert not getattr(right_scaled_row, "_row", False)
        transposed_column_dot = row * jnp.asarray([[1.0], [2.0], [3.0]])
        assert isinstance(transposed_column_dot, Spherefun)
        assert float((transposed_column_dot - 6 * f).norm()) < tol
        with pytest.raises(ValueError, match="mtimes:double2"):
            jnp.asarray([[1.0, 2.0, 3.0]]) * row
        # Pinned MATLAB enters spherefun(store), then errors in its constructor.
        with pytest.raises(ValueError, match="MATLAB:feval:argMustBeStringOrHandle"):
            Spherefunv.mtimes(jnp.asarray([[1.0, 0.0, 0.0],
                                           [0.0, 1.0, 0.0]]), F)

    def test_supplemental_sampled_operator_control(self):
        f = _sph(lambda x, y, z: jnp.sin(y * z) + jnp.cos(2 * x * y))
        F = Spherefunv(f, f, f)
        G = Spherefunv(2 * f, 2 * f, 2 * f)
        H = Spherefunv(f / 2, f / 2, f / 2)
        assert _sampled_vnorm(2 * F - G) < 1e3 * np.finfo(np.float64).eps
        assert _sampled_vnorm(F * 2 - G) < 1e3 * np.finfo(np.float64).eps
        assert _sampled_vnorm(F / 2 - H) < 1e3 * np.finfo(np.float64).eps

    def test_supplemental_array_times_order_and_matmul_orientation(self):
        f = _sph(lambda x, y, z: 1 + x - 2 * y + z)
        F = Spherefunv(f, 2 * f, 3 * f)
        tol = 1e3 * ChebfunPref().cheb2Prefs.chebfun2eps
        col_array = jnp.asarray([[2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        row_array = jnp.asarray([[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]])

        col_result = F.times(col_array)
        assert not getattr(col_result, "_row", False)
        assert float((col_result - Spherefunv(2 * f, 6 * f, 12 * f)).norm()) < tol
        np.testing.assert_array_equal(
            np.asarray(col_result.components[0].pivots),
            np.asarray(jnp.asarray(F.components[0].pivots) / 2.0))

        row = F.transpose()
        row_result = row.times(row_array)
        assert getattr(row_result, "_row", False)
        assert float((row_result - Spherefunv(2 * f, 6 * f, 12 * f).transpose()).norm()) < tol
        # The native left-operand swap preserves multiplication semantics and
        # the row template when the numeric array is on the left.
        swapped = Spherefunv.times(row_array, row)
        assert getattr(swapped, "_row", False)
        assert float((swapped - row_result).norm()) < tol

        with pytest.raises(ValueError, match="times:double"):
            F.times(row_array)

        # MATLAB numel==1 accepts scalar-shaped and 1x1 numeric arrays. All
        # routes must use the scalar factor update rather than reconstruction.
        base = np.asarray(F.components[0].pivots)
        for scalar_array in (jnp.asarray([[3.0]]), jnp.asarray([3.0])):
            product = F.times(scalar_array)
            np.testing.assert_array_equal(
                np.asarray(product.components[0].pivots),
                np.asarray(jnp.asarray(base) / 3.0))
        mtimes_product = F.mtimes(jnp.asarray([[2.0]]))
        np.testing.assert_array_equal(
            np.asarray(mtimes_product.components[0].pivots),
            np.asarray(jnp.asarray(base) / 2.0))

        zero_product = F.times(jnp.asarray([[0.0]]))
        zero_component = zero_product.components[0]
        assert zero_component.pivots.shape == (1,)
        assert bool(jnp.isinf(zero_component.pivots[0]))
        assert len(zero_component.cols) == len(zero_component.rows) == 1
        assert zero_component.idx_plus == F.components[0].idx_plus
        assert zero_component.idx_minus == F.components[0].idx_minus
        assert zero_component.pivot_locations == F.components[0].pivot_locations[:1]
        assert zero_component.nonzero_poles == F.components[0].nonzero_poles
        assert float(jnp.max(jnp.abs(jnp.asarray(zero_component(0.37, 1.11))))) == 0.0

        # ``@`` follows MATLAB mtimes dimensions; only row-by-column contracts.
        with pytest.raises(ValueError, match="mtimes:sizes"):
            F @ F
        contracted = row @ F
        assert isinstance(contracted, Spherefun)
        assert float((contracted - F.dot(F)).norm()) < tol

        # Scalar mtimes propagates orientation according to MATLAB's transpose
        # rewrites: numeric-left keeps a row, numeric-right returns a column.
        assert getattr((2 * row), "_row", False)
        assert not getattr((row * 2), "_row", False)

    def test_supplemental_scalar_division_factor_paths(self):
        f = _sph(lambda x, y, z: 1 + x * y + 2 * z + x ** 2)
        F = Spherefunv(f, 2 * f, 3 * f)
        base = np.asarray(F.components[0].pivot_values)

        # mrdivide takes the literal separableApprox branch
        # pivotValues = pivotValues * divisor.
        for divisor in (3, jnp.asarray([[3.0]])):
            mr = F.mrdivide(divisor)
            np.testing.assert_array_equal(
                np.asarray(mr.components[0].pivot_values),
                np.asarray(jnp.asarray(base) * 3.0))

        # rdivide and scalar-left ldivide/mldivide all reach times(f,1/g),
        # which divides pivots by the computed reciprocal. Keep this separate
        # from the direct mrdivide factor update: the source paths round apart.
        reciprocal = 1 / 3
        # Host binary64 oracle avoids XLA reciprocal-strength reduction.
        expected = np.divide(base, reciprocal)
        outputs = (('rdivide_scalar', F.rdivide(3)),
                   ('rdivide_1x1', F.rdivide(jnp.asarray([[3.0]]))),
                   ('ldivide_scalar', Spherefunv.ldivide(3, F)),
                   ('mldivide_scalar', Spherefunv.mldivide(3, F)))
        for name, actual in outputs:
            np.testing.assert_array_equal(
                np.asarray(actual.components[0].pivot_values), expected,
                err_msg=name)

        with pytest.raises(ValueError, match="rdivide:divByZero"):
            F.rdivide(0)


def test_frozen_native_scalar_factor_rounding():
    fixture = json.loads((Path(__file__).parents[2] / "fixtures"
                          / "spherefunv_scalar_factor_native_20261014.json").read_text())
    # Scalar operations act independently on the pivot vector. Freeze the
    # captured native pivots on two valid factors; do not compare a new
    # adaptive construction against native factorization here.
    base = _sph(lambda x, y, z: 1 + x * y + 2 * z + x**2)
    assert len(base.cols) >= 2
    f = Spherefun(cols=base.cols[:2], rows=base.rows[:2],
                  pivots=jnp.asarray(fixture["input_pivots"]),
                  idx_plus=(0, 1), idx_minus=(),
                  pivot_locations=base.pivot_locations[:2],
                  nonzero_poles=base.nonzero_poles)
    vector = Spherefunv(f, f, f)
    for scalar in (3.0, jnp.asarray(3.0), jnp.asarray([[3.0]])):
        outputs = ((vector.mrdivide(scalar), "mrdivide_by_3"),
                   (vector.rdivide(scalar), "rdivide_by_3"),
                   (Spherefunv.mldivide(scalar, vector), "rdivide_by_3"),
                   (Spherefunv.ldivide(scalar, vector), "rdivide_by_3"))
        for actual, key in outputs:
            np.testing.assert_array_equal(np.asarray(actual.components[0].pivots),
                                          np.asarray(fixture[key]))
