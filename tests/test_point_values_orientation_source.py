"""Source breakpoint storage is independent of Chebfun orientation.

Provenance: @chebfun/getValuesAtBreakpoints.m, transpose.m, restrict.m,
Chebfun7574c77680d7e82b79626300bf255498271a72df. Source-derived controls.
"""
import jax
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.domain import Domain
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2

TECHS = [Chebtech1, Chebtech2]
EPS = float(jnp.finfo(jnp.float64).eps)


def affine(Tech, columns, row):
    coeffs = jnp.asarray([[.5, -.25, 0.], [1., 1., 0.]])[:, :columns]
    if columns == 1:
        coeffs = coeffs[:, 0]
    f = Chebfun(funs=[_Piece(tech=Tech(coeffs=coeffs), interval=(-1., 1.))],
                domain=Domain((-1., 1.)))
    return f.T if row else f


def exact_affine(x, columns):
    x = jnp.asarray(x)
    values = jnp.stack((x+.5, x-.25, jnp.zeros_like(x)), axis=-1)[..., :columns]
    return values[..., 0] if columns == 1 else values


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('columns', [1, 3])
def test_implicit_storage_and_public_evaluation_orientation(Tech, row, columns):
    f = affine(Tech, columns, row)
    expected = exact_affine([-1., 1.], columns)
    assert f.point_values.shape == expected.shape
    assert bool(jnp.array_equal(f.point_values, expected))
    evaluated = f(jnp.asarray([-1., 1.]))
    oriented = expected.T if row and columns > 1 else expected
    assert evaluated.shape == oriented.shape
    assert bool(jnp.array_equal(evaluated, oriented))


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('columns', [2, 3])
def test_public_restrict_square_and_asymmetric_storage(Tech, row, columns):
    f = affine(Tech, columns, row)
    breaks = (-1., -.5, .25, 1.)
    restricted = f.restrict(breaks)
    expected = exact_affine(breaks, columns)
    assert restricted.is_transposed == row
    assert restricted.domain.breakpoints == breaks
    assert restricted.point_values.shape == (4, columns)
    # Original breakpoint rows are copied exactly; square2x2 catches silent
    # axis confusion, asymmetric2x3 catches broadcast failure.
    assert bool(jnp.array_equal(restricted.point_values[jnp.asarray([0, 3])],
                               expected[jnp.asarray([0, 3])]))
    assert float(jnp.max(jnp.abs(restricted.point_values-expected))) < 10*EPS
    values = restricted(jnp.asarray(breaks))
    assert float(jnp.max(jnp.abs(values-(expected.T if row else expected)))) < 10*EPS


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('row', [False, True])
def test_explicit_complex_metadata_is_not_transposed(Tech, row):
    f = affine(Tech, 3, False)
    real = jnp.asarray([[0., -0., 7.], [8., 9., -0.]])
    imag = jnp.asarray([[-0., 2., 3.], [4., 0., 6.]])
    values = jax.lax.complex(real, imag)
    f = f.set_point_values(values)
    f = f.T if row else f
    assert f.point_values.shape == (2, 3)
    assert jax.device_get(f.point_values).tobytes() == jax.device_get(values).tobytes()
    restricted = f.restrict((-1., 0., 1.))
    actual = restricted.point_values[jnp.asarray([0, 2])]
    assert jax.device_get(actual).tobytes() == jax.device_get(values).tobytes()
    assert restricted.is_transposed == row


@pytest.mark.parametrize('Tech', TECHS)
@pytest.mark.parametrize('row', [False, True])
def test_neighbor_average_and_explicit_jump_survive_restriction(Tech, row):
    pieces = [_Piece(tech=Tech(coeffs=jnp.asarray([values])), interval=interval)
              for values, interval in [([1., 2.], (-1., 0.)),
                                       ([5., 8.], (0., 1.))]]
    f = Chebfun(funs=pieces, domain=Domain((-1., 0., 1.)))
    f = f.T if row else f
    expected = jnp.asarray([[1., 2.], [3., 5.], [5., 8.]])
    assert bool(jnp.array_equal(f.point_values, expected))
    assert bool(jnp.array_equal(f.restrict((-.5, 0., .5)).point_values, expected))
    explicit = expected.at[1].set(jnp.asarray([11., -7.]))
    restricted = f.set_point_values(explicit).restrict((-.5, 0., .5))
    assert bool(jnp.array_equal(restricted.point_values, explicit))
    assert restricted.is_transposed == row


def test_empty_storage_shape():
    assert Chebfun.empty().point_values.shape == (0,)


@pytest.mark.parametrize('Tech', TECHS)
def test_no_restored_breakpoints_keep_real_dtype(Tech):
    f = affine(Tech, 3, True).set_point_values(
        jnp.asarray([[1j, 2j, 3j], [4j, 5j, 6j]]))
    restricted = f.restrict((-.5, .25))
    assert restricted.point_values.dtype == jnp.float64
    reference = affine(Tech, 3, True).restrict((-.5, .25))
    assert (jax.device_get(restricted.point_values).tobytes()
            == jax.device_get(reference.point_values).tobytes())
    assert float(jnp.max(jnp.abs(restricted.point_values
                                - exact_affine([-.5, .25], 3)))) < 10*EPS
