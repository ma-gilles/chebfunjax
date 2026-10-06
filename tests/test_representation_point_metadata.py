"""Simplification/prolongation must preserve genuine point discontinuities."""

import jax.numpy as jnp

# uses-numpy: independent exact metadata assertions only.
import numpy as np
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.operators.chebop import Chebop, SystemSolution, _commonize_system
from chebfunjax.tech.chebtech import Chebtech2


def make_field(coefficients, points):
    piece = _Piece(Chebtech2.from_coeffs(jnp.asarray(coefficients)), (-1., 1.))
    return Chebfun(funs=[piece], domain=Domain((-1., 1.))).set_point_values(jnp.asarray(points))


def test_simplify_preserves_point_discontinuities_without_changing_limits():
    field = make_field([2., 1., 0., 0.], [17., -9.])
    simplified = field.simplify()
    np.testing.assert_array_equal(simplified(jnp.asarray([-1., 1.])), [17., -9.])
    # The polynomial limits must remain 1 and 3, not be adjusted to metadata.
    np.testing.assert_allclose(simplified.funs[0](jnp.asarray([-1., 1.])), [1., 3.], rtol=0, atol=1e-15)
    np.testing.assert_allclose(simplified(jnp.asarray([-.5, .25])), [1.5, 2.25], rtol=0, atol=1e-15)


def test_commonize_preserves_each_components_stored_data_and_polynomial():
    fields = [make_field([2., 1.], [17., -9.]), make_field([1., 0., 2., 1.], [30., 40.])]
    outputs = _commonize_system(fields)
    assert len(outputs[0]) == len(outputs[1])
    for before, after in zip(fields, outputs):
        np.testing.assert_array_equal(after.point_values, before.point_values)
        np.testing.assert_array_equal(after(jnp.asarray([-1., 1.])), before.point_values)
        np.testing.assert_allclose(after(jnp.asarray([-.5, .25])), before(jnp.asarray([-.5, .25])), rtol=0, atol=1e-15)


def test_actual_solution_finalizer_preserves_metadata_across_both_stages():
    fields = SystemSolution([make_field([2., 1.], [17., -9.]), make_field([1., 0., 2., 1.], [30., 40.])])
    outputs = Chebop._simplify_solution(fields)
    assert isinstance(outputs, SystemSolution)
    for before, after in zip(fields, outputs):
        np.testing.assert_array_equal(after.point_values, before.point_values)


@pytest.mark.parametrize("transposed", [False, True])
def test_simplify_preserves_complex_multicolumn_metadata_and_orientation(transposed):
    coefficients = jnp.asarray([[2.+1j, 4.-2j], [1., -2.], [0., 0.]])
    points = jnp.asarray([[17.+3j, -9.-2j], [30.-1j, 40.+4j]])
    piece = _Piece(Chebtech2.from_coeffs(coefficients), (-1., 1.))
    field = Chebfun(funs=[piece], domain=Domain((-1., 1.))).set_point_values(points)
    if transposed:
        field = field.T
    simplified = field.simplify()
    assert simplified.is_transposed is transposed
    np.testing.assert_array_equal(simplified.point_values, points)
    np.testing.assert_array_equal(simplified._point_values, field._point_values)
    np.testing.assert_allclose(simplified.funs[0](jnp.asarray([-.5, .25])),
                               field.funs[0](jnp.asarray([-.5, .25])), rtol=0, atol=1e-15)
