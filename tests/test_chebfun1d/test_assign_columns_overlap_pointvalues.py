"""Stored versus side-limit values through native overlap/assignColumns.

Provenance: @chebfun/{overlap,restrict,assignColumns}.m, Chebfun7574c77.
Inherited pointValues are exact stored metadata. Newly inserted values use
smooth FUN limits; the tiny linear/constant degree permits a16eps*3 roundoff
budget for representation evaluation, separate from exact metadata checks.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


@pytest.mark.parametrize("tech", [Chebtech1, Chebtech2])
def test_assignment_overlap_preserves_inherited_values_and_new_limits(tech):
    f = Chebfun(funs=[_Piece(tech=tech.from_coeffs(jnp.array([value])), interval=interval)
                     for value, interval in [(1., (-1., 0.)), (3., (0., 1.))]],
                domain=Domain((-1., 0., 1.)))
    f = f.set_point_values(jnp.array([11., 12., 13.]))
    intervals = [(-1., -.5), (-.5, .5), (.5, 1.)]
    g = Chebfun(funs=[_Piece(tech=tech.from_coeffs(jnp.array([(a+b)/2, (b-a)/2])),
                            interval=(a, b)) for a, b in intervals],
                domain=Domain((-1., -.5, .5, 1.)))
    g = g.set_point_values(jnp.array([21., 22., 23., 24.]))
    result = f.assign_columns(1, g)
    assert tuple(result.domain.breakpoints) == (-1., -.5, 0., .5, 1.)
    expected = jnp.array([[11., 21.], [1., 22.], [12., 0.], [3., 23.], [13., 24.]])
    values = result.point_values
    assert jnp.array_equal(values[jnp.array([0, 2, 4]), 0], jnp.array([11., 12., 13.]))
    assert jnp.array_equal(values[jnp.array([0, 1, 3, 4]), 1], jnp.array([21., 22., 23., 24.]))
    bound = 16*jnp.finfo(jnp.float64).eps*3
    assert jnp.max(jnp.abs(values-expected)) < bound
    assert jnp.max(jnp.abs(result.funs[1](jnp.asarray(0.))-jnp.array([1., 0.]))) < bound
    assert jnp.max(jnp.abs(result.funs[2](jnp.asarray(0.))-jnp.array([3., 0.]))) < bound
    assert all(type(piece.tech) is tech for piece in result.funs)
    assert jnp.array_equal(f.point_values, jnp.array([11., 12., 13.]))
    assert jnp.array_equal(g.point_values, jnp.array([21., 22., 23., 24.]))
