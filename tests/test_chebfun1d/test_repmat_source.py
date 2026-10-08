"""Source repmat/horzcat/vertcat metadata, shape and representation controls.

Provenance: Chebfun7574c77680d7e82b79626300bf255498271a72df.
"""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, chebfun
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.operators.chebmatrix import ChebMatrix


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('form', ['pair', 'vector', 'adapter'])
def test_point_values_orientation_and_coefficients(row, form):
    f = chebfun(jnp.sin, domain=(-1, 0, 1)).set_point_values(jnp.array([7., 8., 9.]))
    if row:
        f = f.T
    shape = (3, 1) if row else (1, 3)
    args = shape if form == 'pair' else (list(shape),) if form == 'vector' else (3,)
    out = f.repmat(*args)
    assert out.is_transposed == row
    assert out.domain == f.domain
    assert jnp.array_equal(out.point_values, jnp.array([[7., 7., 7.], [8., 8., 8.], [9., 9., 9.]]))
    for before, after in zip(f.funs, out.funs):
        assert type(after.tech) is type(before.tech)
        assert jnp.array_equal(after.tech.coeffs, jnp.tile(before.tech.coeffs[:, None], (1, 3)))
    assert jnp.array_equal(f.point_values, jnp.array([7., 8., 9.]))


def test_array_columns_tile_in_source_order():
    f = chebfun(lambda x: jnp.stack([x, x**2], axis=-1), domain=(-1, 0, 1))
    f = f.set_point_values(jnp.array([[1., 2.], [3., 4.], [5., 6.]]))
    q = f.repmat(1, 3)
    x = jnp.array([-.6, .3])
    assert jnp.max(jnp.abs(q(x)-jnp.tile(f(x), (1, 3)))) < 1e-14
    assert jnp.array_equal(q.point_values, jnp.tile(f.point_values, (1, 3)))


def test_array_row_uses_source_chebmatrix_promotion():
    f = chebfun(lambda x: jnp.stack([x, x**2], axis=-1)).T
    q = f.repmat(2, 1)
    assert isinstance(q, ChebMatrix)
    assert (q.nrows, q.ncols) == (4, 1)
    x = jnp.array([-.6, .3])
    for i, row in enumerate(q.blocks):
        assert row[0].is_transposed
        assert jnp.max(jnp.abs(row[0](x)-x**(i % 2+1))) < 1e-14


@pytest.mark.parametrize('row', [False, True])
def test_periodic_representation_is_retained(row):
    f = chebfun(jnp.sin, domain=(-jnp.pi, jnp.pi), trig=True)
    if row:
        f = f.T
    q = f.repmat(2)
    assert q.is_transposed == row
    assert type(q.funs[0].tech) is type(f.funs[0].tech)
    x = jnp.array([-.7, .2])
    actual = q(x).T if row else q(x)
    assert jnp.max(jnp.abs(actual-jnp.stack([jnp.sin(x)]*2, axis=-1))) < 1e-14


def test_delta_column_keeps_separate_quasimatrix_columns():
    f = chebfun(lambda x: x).set_point_values(jnp.array([7., 9.]))
    f = Chebfun(funs=f.funs, domain=f.domain, deltas=((0., 2., 0),)).set_point_values(f.point_values)
    q = f.repmat(1, 2)
    assert isinstance(q, Quasimatrix)
    assert all(c.deltas == f.deltas for c in q.cols)
    assert all(jnp.array_equal(c.point_values, f.point_values) for c in q.cols)


@pytest.mark.parametrize('row,args', [(False, (2, 1)), (True, (1, 2)), (False, (1, 2.5)), (False, ([1, 2, 3],))])
def test_source_invalid_tiling_rejected(row, args):
    f = chebfun(jnp.sin)
    if row:
        f = f.T
    with pytest.raises(ValueError):
        f.repmat(*args)


def test_one_copy_preserves_scalar_metadata():
    f = chebfun(jnp.sin).set_point_values(jnp.array([7., 9.])).T
    assert f.repmat(1, 1) is f


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('count', [0, -1])
def test_nonpositive_count_returns_source_numeric_empty(row, count):
    f = chebfun(jnp.sin)
    if row:
        f = f.T
    q = f.repmat(count)
    assert isinstance(q, jnp.ndarray) and q.shape == (0, 0)


def test_singular_columns_keep_source_representation():
    from chebfunjax.chebfun1d.chebfun import _Piece
    from chebfunjax.domain import Domain
    from chebfunjax.fun.singfun import Singfun
    from chebfunjax.tech.chebtech import Chebtech2
    tech = Singfun(Chebtech2.from_coeffs(jnp.array([1., .2])), (-.3, -.2))
    f = Chebfun(funs=[_Piece(tech=tech, interval=(-1., 1.))], domain=Domain((-1., 1.)))
    q = f.repmat(1, 2)
    assert isinstance(q, Quasimatrix)
    assert all(c.funs[0].tech is tech for c in q.cols)
