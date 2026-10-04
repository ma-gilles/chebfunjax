"""Focused MATLAB contracts for empty and ``any`` Trigtech behavior.

Provenance
----------
MATLAB source: @trigtech/trigtech.m, isempty.m, size.m, vscale.m,
    any.m, cumsum.m, horzcat.m; tests/trigtech/test_isempty.m,
    test_any.m.
Chebfun commit: 7574c77.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from chebfunjax.tech.trigtech import Trigtech


class TestTrigtechEmptySourceContracts:
    def test_default_empty_storage_shape_size_and_scale(self):
        # MATLAB no-argument construction leaves values/coeffs as [] (0x0).
        f = Trigtech()
        assert f.isempty()
        assert f.coeffs.shape == (0, 0)
        assert f.values.shape == (0, 0)
        assert f.size() == (0, 0)
        assert f.size(1) == 0
        assert f.size(2) == 0
        assert f.vscale == 0.0

    def test_zero_column_empty_keeps_source_size_shape(self):
        # An empty 3-by-0 values array has zero elements, but size(f) still
        # returns the stored MATLAB values shape (3, 0).
        f = Trigtech(coeffs=jnp.empty((3, 0), dtype=jnp.complex128))
        assert f.isempty()
        assert f.values.shape == (3, 0)
        assert f.size() == (3, 0)
        assert f.size(1) == 3
        assert f.size(2) == 0

    def test_zero_polynomial_is_not_empty(self):
        # A one-coefficient zero function is a valid nonempty Trigtech.
        f = Trigtech.from_coeffs(jnp.zeros((1,), dtype=jnp.complex128))
        assert not f.isempty()
        assert f.size() == (1, 1)
        assert f.vscale == 0.0

    def test_empty_cumsum_is_identity(self):
        # @trigtech/cumsum.m returns immediately for isempty(f).
        f = Trigtech()
        out = f.cumsum()
        assert out.isempty()
        assert out.coeffs.shape == (0, 0)
        assert out.values.shape == (0, 0)
        np.testing.assert_array_equal(np.asarray(out.coeffs), np.asarray(f.coeffs))

    def test_horzcat_drops_empty_inputs(self):
        # @trigtech/horzcat.m removes empties before concatenating.
        empty = Trigtech()
        f = Trigtech.from_coeffs(jnp.array([2.0 + 0.0j]))
        out = Trigtech.horzcat(empty, f, empty)
        np.testing.assert_array_equal(np.asarray(out.coeffs), np.asarray(f.coeffs))
        assert not out.isempty()

    def test_horzcat_all_empty_returns_first_empty(self):
        # Source returns varargin{1} when all input techs are empty.
        first = Trigtech(coeffs=jnp.empty((3, 0), dtype=jnp.complex128))
        out = Trigtech.horzcat(first, Trigtech())
        assert out.isempty()
        assert out.coeffs.shape == (3, 0)


class TestTrigtechAnySourceContracts:
    def test_default_empty_any_is_false(self):
        # tests/trigtech/test_any.m pass(1): ~any(testclass), with testclass
        # constructed by trigtech() and therefore values=[] (0x0).
        assert bool(Trigtech().any()) is False

    def test_empty_any_explicit_dim1_is_false(self):
        # @trigtech/any.m dim=1 branch also invokes any(f.values).
        assert bool(Trigtech().any(dim=1)) is False

    def test_any_down_columns_ignores_nan_values(self):
        # Source any.m calls MATLAB any(f.values), which ignores NaNs (also
        # documented at https://www.mathworks.com/help/matlab/ref/any.html).
        # The first column is nonzero complex, the second zero, and the third
        # NaN-only. Three sample rows force default reduction along dimension 1.
        coeffs = jnp.array(
            [[0.0 + 0.0j, 0.0 + 0.0j, jnp.nan + 0.0j],
             [0.0 + 1.0j, 0.0 + 0.0j, jnp.nan + 0.0j],
             [0.0 + 0.0j, 0.0 + 0.0j, jnp.nan + 0.0j]],
            dtype=jnp.complex128,
        )
        f = Trigtech.from_coeffs(coeffs, is_real=False)
        got = np.asarray(f.any())
        np.testing.assert_array_equal(got, np.array([True, False, False]))

    def test_any_is_jittable(self):
        f = Trigtech.from_coeffs(
            jnp.array([[0.0 + 0.0j], [1.0 + 0.0j]], dtype=jnp.complex128),
            is_real=False,
        )
        got = jax.jit(lambda tech: tech.any(dim=1))(f)
        np.testing.assert_array_equal(np.asarray(got), np.array([True]))

    def test_any_single_sample_uses_first_nonsingleton_dimension(self):
        # MATLAB any(1xM) reduces across columns, rather than returning M flags.
        f = Trigtech.from_coeffs(
            jnp.array([[0.0 + 0.0j, 2.0 + 0.0j]], dtype=jnp.complex128),
            is_real=True,
        )
        assert f.values.shape == (1, 2)
        assert bool(f.any()) is True
        assert bool(f.any(dim=1)) is True

    def test_any_empty_across_columns_returns_zero_constant(self):
        # feval(empty, arbitraryPoint) is [], and source any([]) is false;
        # dim=2 stores that result into values/coeffs of the output tech.
        got = Trigtech().any(dim=2)
        assert not got.isempty()
        np.testing.assert_array_equal(np.asarray(got.coeffs), np.array([[False]]))

    def test_any_across_columns_returns_constant_tech(self):
        # test_any.m pass(3): at the source arbitrary point at least one
        # component is nonzero, so the returned constant has coefficient 1.
        coeffs = jnp.array([[1.0 + 0.0j, 0.0 + 0.0j, 2.0 + 0.0j]])
        f = Trigtech.from_coeffs(coeffs, is_real=True)
        got = f.any(dim=2)
        np.testing.assert_array_equal(np.asarray(got.coeffs), np.array([[True]]))

    def test_invalid_dimension_has_source_identifier(self):
        f = Trigtech.from_coeffs(jnp.array([1.0 + 0.0j]))
        try:
            f.any(dim=3)
        except ValueError as exc:
            assert "TRIGTECH:any:dim" in str(exc)
        else:
            raise AssertionError("any(dim=3) must reject unsupported dimensions")

    def test_any_across_columns_of_zero_array_is_zero(self):
        # test_any.m pass(4): any([0*x 0*x], 2) is the zero constant tech.
        f = Trigtech.from_coeffs(jnp.zeros((1, 2), dtype=jnp.complex128))
        got = f.any(dim=2)
        np.testing.assert_array_equal(np.asarray(got.coeffs), np.array([[False]]))
