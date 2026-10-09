"""MATLAB cell(1,n) accepts a numeric scalar regardless of singleton shape.

Source: @chebfun/vander.m at7574c77680d7e82b79626300bf255498271a72df.
These controls qualify only the new Python dimension adapter.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech2


def linear():
    return Chebfun(funs=[_Piece(tech=Chebtech2(coeffs=jnp.asarray([.25, .5])),
                                interval=(-1., 1.))], domain=Domain((-1., 1.)))


@pytest.mark.parametrize('shape', [(1,), (1, 1)])
@pytest.mark.parametrize('count', [-1, 0, 1, 2])
def test_single_numeric_dimension_matches_scalar(shape, count):
    f = linear()
    actual = f.vander(jnp.asarray(count).reshape(shape))
    reference = f.vander(count)
    assert actual.n_columns == reference.n_columns == max(count, 1)
    assert actual.domain == reference.domain
    assert bool(jnp.array_equal(actual.coeffs, reference.coeffs))
    assert bool(jnp.array_equal(actual.point_values, reference.point_values))


@pytest.mark.parametrize('shape', [(2,), (2, 1)])
def test_multiple_numeric_dimensions_rejected(shape):
    with pytest.raises(ValueError, match='column count must be a scalar'):
        linear().vander(jnp.asarray([1, 2]).reshape(shape))


@pytest.mark.parametrize('count', [.5, float('inf'), float('nan')])
def test_finite_integer_dimension_predicates_unchanged(count):
    with pytest.raises(ValueError, match='finite integer'):
        linear().vander(jnp.asarray([count]))
