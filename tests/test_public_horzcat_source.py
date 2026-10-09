"""Focused public adapters; native nine-slot suite is a separate gate."""
import jax.numpy as jnp
import pytest

from chebfunjax.chebfun1d.chebfun import Chebfun, _Piece
from chebfunjax.chebfun1d.linalg import Quasimatrix
from chebfunjax.domain import Domain
from chebfunjax.operators.chebmatrix import ChebMatrix
from chebfunjax.tech.chebtech import Chebtech1, Chebtech2


def function(tech=Chebtech2, breaks=(-1., 0., 1.), complex_data=False):
    c = jnp.array([1., .25]) * (1+2j if complex_data else 1.)
    return Chebfun(funs=[_Piece(tech=tech(coeffs=c, ishappy=False),
                               interval=(a, b))
                        for a, b in zip(breaks[:-1], breaks[1:])],
                   domain=Domain(breaks))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
@pytest.mark.parametrize('complex_data', [False, True])
def test_numeric_full_domain(tech, complex_data):
    f = function(tech, complex_data=complex_data)
    row = jnp.array([2., 3.]) * (1+1j if complex_data else 1.)
    q = Chebfun.horzcat([row, f])
    assert isinstance(q, Chebfun) and q.n_columns == 3
    assert q.domain == f.domain
    assert all(isinstance(p.tech, Chebtech2) for p in q.funs)
    assert all(p.tech.ishappy for p in q.funs)
    for p in q.funs:
        assert bool(jnp.array_equal(p.tech.coeffs[0, :2], row))
        assert bool(jnp.all(p.tech.coeffs[1:, :2] == 0))
    assert bool(jnp.array_equal(q.point_values[:, :2],
                               jnp.broadcast_to(row, (3, 2))))


@pytest.mark.parametrize('tech', [Chebtech1, Chebtech2])
def test_list_variadic_and_metadata(tech):
    f = function(tech)
    pv = jnp.array([7., 8., 9.])
    f = f.set_point_values(pv)
    q = Chebfun.horzcat(f, 2.)
    r = Chebfun.horzcat([f, 2.])
    assert all(isinstance(p.tech, tech) and not p.tech.ishappy for p in q.funs)
    assert bool(jnp.array_equal(q.point_values[:, 0], pv))
    assert bool(jnp.array_equal(q.point_values, r.point_values))


def test_empty_and_singleton():
    f = function()
    e = Chebfun.empty()
    assert Chebfun.horzcat([]).isempty()
    assert Chebfun.horzcat([e, e]) is e
    assert Chebfun.horzcat([e, f, []]) is f
    assert Chebfun.horzcat([f]) is f
    with pytest.raises(ValueError, match='requires a Chebfun'):
        Chebfun.horzcat([1., 2.])


def test_orientation_and_endpoint_error():
    f = function()
    with pytest.raises(ValueError, match='horzcat:transpose'):
        Chebfun.horzcat([f, f.T])
    with pytest.raises(ValueError, match='horzcat:domains'):
        Chebfun.horzcat([f, function(breaks=(0., 1.))])


def test_quasimatrix_flatten_order():
    f = function()
    g = function(breaks=(-1., .5, 1.))
    q = Chebfun.horzcat([f, g])
    assert isinstance(q, Quasimatrix)
    r = Chebfun.horzcat([q, f])
    assert r.cols == [f, g, f]
    a = Chebfun.horzcat([f, f])
    r = Chebfun.horzcat([a, g])
    assert isinstance(r, Quasimatrix) and len(r.cols) == 3
    assert r.cols[-1] is g


@pytest.mark.parametrize('complex_data', [False, True])
@pytest.mark.parametrize('array_input', [False, True])
def test_row_blocks(complex_data, array_input):
    f = function(complex_data=complex_data)
    if array_input:
        f = Chebfun.horzcat([f, f])
    row = jnp.array([0., 2.]) * (1+3j if complex_data else 1.)
    with pytest.warns(UserWarning, match='vertcat:join'):
        q = Chebfun.horzcat([f.T, row])
    assert isinstance(q, ChebMatrix)
    assert (q.nrows, q.ncols) == (1, f.n_columns+2)
    assert q.domain == f.domain.breakpoints
    for b in q.blocks[0][:-2]:
        assert b.is_transposed
        assert bool(jnp.array_equal(b.funs[0].tech.coeffs,
                                   jnp.conj(function(complex_data=complex_data)
                                            .funs[0].tech.coeffs)))
    assert bool(jnp.array_equal(jnp.asarray(q.blocks[0][-2:]), jnp.conj(row)))


def test_row_scalar_and_domain_merge():
    f = function()
    g = function(breaks=(-1., .5, 1.))
    q = Chebfun.horzcat([f.T, g.T, 2j])
    assert q.domain == (-1., 0., .5, 1.)
    assert q.blocks[0][-1] == -2j
    with pytest.raises(ValueError, match='sizeMismatch'):
        Chebfun.horzcat([f.T, jnp.ones((2, 1))])
    with pytest.raises(ValueError, match='merge:incompat'):
        Chebfun.horzcat([f.T, function(breaks=(0., 2.)).T])


@pytest.mark.parametrize('shape', [(2, 1), (2, 2), (1, 2)])
def test_numeric_matrix_interval_reuse(shape):
    f = function()
    values = jnp.arange(shape[0]*shape[1], dtype=jnp.float64).reshape(shape)+1j
    q = Chebfun.horzcat([values, f])
    expected = Chebtech2.from_values(values).coeffs
    for piece in q.funs:
        assert bool(jnp.array_equal(piece.tech.coeffs[:expected.shape[0], :shape[1]], expected))


def test_literal_row_domain_merge_near_breaks():
    eps = float(jnp.finfo(jnp.float64).eps)
    f = function()
    g = function(breaks=(-1., 10*eps, 1.))
    q = Chebfun.horzcat([f.T, g.T])
    assert q.domain == (-1., 0., 1.)
