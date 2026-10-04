"""MATLAB MTIMES / TIMES and empty concatenation source contracts.

Provenance
----------
MATLAB source : @chebtech/{mtimes,times,isempty,horzcat,cell2mat}.m
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
Independent controls use exact Chebyshev coefficients and dyadic samples.
"""

import jax
import jax.numpy as jnp
import pytest

from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_mtimes_rejects_left_vector_even_if_coefficient_length_matches(tech):
    f = tech.from_coeffs(jnp.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError) as exc:
        [2, 3, 4] @ f
    assert exc.value.identifier == "CHEBFUN:CHEBTECH:mtimes:size"
    assert str(exc.value) == "Inner matrix dimensions must agree."


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_numeric_left_times_keeps_source_column_scaling(tech):
    coefficients = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    f = tech.from_coeffs(coefficients)
    # MATLAB permits row-vector double .* CHEBTECH by recursion to TIMES.
    left = [2, -3] * f
    right = f * [2, -3]
    expected = jnp.array([[2.0, -6.0], [6.0, -12.0], [10.0, -18.0]])
    assert jnp.array_equal(left.coeffs, expected)
    assert jnp.array_equal(right.coeffs, expected)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_mtimes_scalar_singleton_and_explicit_row_column_shapes(tech):
    scalar = tech.from_coeffs(jnp.array([1.0, 2.0]), ishappy=False)
    scaled = scalar @ [[2]]
    assert jnp.array_equal(scaled.coeffs, jnp.array([2.0, 4.0]))
    assert not scaled.ishappy
    assert jnp.array_equal((2 @ scalar).coeffs, scaled.coeffs)

    expanded = scalar @ [[2, 3]]
    assert jnp.array_equal(expanded.coeffs, jnp.array([[2.0, 3.0], [4.0, 6.0]]))
    vector = tech.from_coeffs(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    contracted = vector @ [[2], [-1]]
    assert jnp.array_equal(contracted.coeffs, jnp.array([[0.0], [2.0]]))
    points = jnp.array([-0.5, 0.5])
    assert jnp.array_equal(contracted(points), jnp.array([[-1.0], [1.0]]))


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_zero_matrix_result_collapses_coefficients_and_keeps_columns(tech):
    f = tech.from_coeffs(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    g = f @ jnp.zeros((2, 3), dtype=jnp.float64)
    assert g.coeffs.shape == (1, 3)
    assert jnp.array_equal(g.coeffs, jnp.zeros((1, 3)))


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_empty_coefficients_and_empty_operands_precede_mtimes_type_checks(tech):
    f = tech.from_coeffs(jnp.array([1.0, 2.0]))
    for coefficients in [jnp.empty((0,)), jnp.empty((3, 0))]:
        e = tech.from_coeffs(coefficients)
        assert e.isempty()
        assert (e @ jnp.uint8(128)).isempty()
        assert ([1, 2, 3] @ e).isempty()
        assert (e * f).isempty()
    assert (f @ []).isempty()
    assert ([] @ f).isempty()
    # Additional TIMES controls, distinct from source MTIMES via @.
    assert (f * []).isempty()
    assert ([] * f).isempty()


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_horizontal_concat_filters_empties_and_preserves_single_input_identity(tech):
    f = tech.from_coeffs(jnp.array([1.0, 2.0]))
    e = tech.empty()
    assert tech.cell2mat([e, e]) is e
    assert tech.cell2mat([e, f]) is f
    assert tech.cell2mat([f, e]) is f
    assert tech.cell2mat([f]) is f
    assert tech.cell2mat([[], f]) is f

    h = tech.from_coeffs(jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    result = tech.cell2mat([f, e, h])
    assert jnp.array_equal(result.coeffs, jnp.array([[1.0, 0.0, 1.0],
                                                                 [2.0, 1.0, 0.0],
                                                                 [0.0, 0.0, 1.0]]))
    points = jnp.array([-0.5, 0.5])
    expected = jnp.stack((1 + 2 * points, points, 2 * points**2), axis=1)
    assert jnp.array_equal(result(points), expected)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_horizontal_concat_rejects_nontech_inputs_with_source_diagnostic(tech):
    f = tech.from_coeffs(jnp.array([1.0, 2.0]))
    with pytest.raises(ValueError) as exc:
        tech.cell2mat([f, [1, 2]])
    assert exc.value.identifier == "CHEBFUN:CHEBTECH:horzcat:typeMismatch"
    assert str(exc.value) == "Incompatible concatenation. Ensure discretizations are of the same type."


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_nonzero_matrix_product_can_be_traced_with_jax(tech):
    coefficients = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    matrix = jnp.array([[2.0, 3.0], [4.0, 5.0]])
    actual = jax.jit(lambda c, a: (tech.from_coeffs(c) @ a).coeffs)(coefficients, matrix)
    expected = jnp.array([[10.0, 13.0], [22.0, 29.0]])
    assert jnp.array_equal(actual, expected)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_scalar_tech_times_row_vector_expands_columns(tech):
    f = tech.from_coeffs(jnp.array([1.0, 2.0, 3.0]))
    expected = jnp.array([[2.0, -3.0], [4.0, -6.0], [6.0, -9.0]])
    assert jnp.array_equal((f * [2, -3]).coeffs, expected)
    assert jnp.array_equal(([2, -3] * f).coeffs, expected)


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_times_column_multiplier_follows_source_matrix_contraction(tech):
    f = tech.from_coeffs(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    result = f * [[2], [-1]]
    assert jnp.array_equal(result.coeffs, jnp.array([[0.0], [2.0]]))


def test_chebtech_kinds_share_the_same_coefficient_basis_for_times():
    f = Chebtech1.from_coeffs(jnp.array([1.0, 2.0]))
    g = Chebtech2.from_coeffs(jnp.array([2.0, -1.0]))
    # (1+2x)*(2-x) = 2+3x-2x^2 = 1*T0+3*T1-1*T2.
    expected = jnp.array([1.0, 3.0, -1.0])
    assert jnp.max(jnp.abs((f * g).coeffs - expected)) < 1e-14
    assert jnp.max(jnp.abs((g * f).coeffs - expected)) < 1e-14


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_constant_times_uses_source_numeric_recursion_without_chopping(tech):
    coefficients = jnp.array([1.0, 2.0, 0.0, 0.0])
    f = tech.from_coeffs(coefficients, ishappy=False)
    constant = tech.from_coeffs(jnp.array([2.0]))
    for result in [f * constant, constant * f]:
        assert jnp.array_equal(result.coeffs, 2 * coefficients)
        assert not result.ishappy


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_pointwise_times_dimension_identifier_and_single_column_broadcast(tech):
    f = tech.from_coeffs(jnp.array([[1.0, 2.0], [3.0, 4.0]]))
    g = tech.from_coeffs(jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(ValueError) as exc:
        f * g
    assert exc.value.identifier == "CHEBFUN:CHEBTECH:times:dim2"
    assert str(exc.value) == "Inner matrix dimensions must agree."
    scalar = tech.from_coeffs(jnp.array([[1.0], [2.0]]))
    # (1+2x)*(1+3x) and (1+2x)*(2+4x), in the T basis.
    expected = jnp.array([[4.0, 6.0], [5.0, 8.0], [3.0, 4.0]])
    for result in [f * scalar, scalar * f]:
        assert jnp.max(jnp.abs(result.coeffs - expected)) < 1e-14


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_conjugate_products_enforce_nonnegative_grid_values(tech):
    f = tech.from_coeffs(jnp.array([0.0 + 0.0j, 0.0 + 1.0j]))
    result = f * f.conj()
    expected = jnp.array([0.5, 0.0, 0.5])
    assert jnp.max(jnp.abs(result.coeffs - expected)) < 1e-14
    assert jnp.all(tech.coeffs2vals(result.coeffs) >= 0)
