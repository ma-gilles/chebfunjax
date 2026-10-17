"""Authored GET source-value/type controls, pin7574c77.

No new original MATLAB assertions are claimed. Adaptive constructors retain
their defaults; endpoint/scale tests use source operands and actual coefficient
representations. Metadata happiness control changes only its explicit flag.
"""
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj
from chebfunjax.chebfun1d.chebfun import get
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.domain import _linear_forward_map
from chebfunjax.tech.chebtech import Chebtech1
from chebfunjax.utils.quadrature import chebpts


@pytest.mark.parametrize('ends', [(.2, .5), (-.2, .1), (-2., 3.)])
def test_source_hscale_bounded(ends):
    f = cj.chebfun(lambda x: x, domain=ends)
    assert get(f, 'hscale') == max(abs(ends[0]), abs(ends[1]))


def test_source_hscale_unbounded():
    f = cj.chebfun(lambda x: jnp.exp(-x), domain=(0., float('inf')))
    assert get(f, 'hscale') == 1.


@pytest.mark.parametrize(('array', 'complex_values', 'row', 'override'), [
    (False, False, False, False), (False, False, True, False),
    (True, False, False, False), (True, True, True, False),
    (True, True, False, True),
])
def test_source_pointvalues_raw_field(array, complex_values, row, override):
    def op(x):
        y = x + (1j*x*x if complex_values else 0.)
        return jnp.stack((y, 2*y), axis=-1) if array else y
    f = cj.chebfun(op, domain=(2., 3., 5.))
    if override:
        f = f.set_point_values(jnp.array([[9.+2j, 8.-3j], [7.+4j, 6.-5j], [5.+6j, 4.-7j]]))
    if row:
        f = f.T
    before = f.point_values
    actual = get(f, 'pointValues')
    expected = jnp.asarray(before).reshape((3, -1))
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(f.point_values, before)


def test_source_empty_pointvalues_field():
    out = get(cj.chebfun(), 'pointValues')
    assert out.shape == (0, 0) and out.dtype == jnp.float64


@pytest.mark.parametrize('unhappy', [False, True])
def test_source_ishappy_all_actual_piece_flags(unhappy):
    f = cj.chebfun(lambda x: x*x, domain=(2., 3., 5.))
    if unhappy:
        pieces = list(f.funs)
        piece = pieces[-1]
        tech = type(piece.tech)(coeffs=piece.tech.coeffs, ishappy=False)
        pieces[-1] = type(piece)(tech=tech, interval=piece.interval)
        f = cj.Chebfun(funs=pieces, domain=f.domain)
    out = get(f, 'ishappy')
    assert bool(out) is (not unhappy)
    assert isinstance(out, (bool, jnp.ndarray))


def test_empty_happiness_remains_explicitly_unqualified():
    # Native empty loops leave output unassigned; exact ID is not captured.
    with pytest.raises(NotImplementedError):
        get(cj.chebfun(), 'ishappy')


@pytest.fixture(scope='module', params=['chebtech1', 'chebtech2'])
def f(request):
    def op(x):
        return jnp.stack((x+1j*x*x, 100*x+2-2j*x), axis=-1)
    return cj.chebfun(op, domain=(2., 3., 5.), pref=ChebfunPref(tech=request.param))


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('prop', ['lval-local', 'rval-local', 'vscale-local', 'points'])
def test_actual_finite_tech_source_leaf_values(f, row, prop):
    obj = f.T if row else f
    actual = get(obj, prop)
    leaves = []
    for piece in f.funs:
        tech = piece.tech
        c = tech.coeffs
        c = c[:, None] if c.ndim == 1 else c
        if prop == 'lval-local':
            value = jnp.sum(c * ((-1.)**jnp.arange(c.shape[0]))[:, None], axis=0, keepdims=True)
        elif prop == 'rval-local':
            value = jnp.sum(c, axis=0, keepdims=True)
        elif prop == 'vscale-local':
            values = tech.values
            values = values[:, None] if values.ndim == 1 else values
            value = jnp.max(jnp.abs(values), axis=0, keepdims=True)
        else:
            points = _linear_forward_map(
                chebpts(len(tech), kind=1 if isinstance(tech, Chebtech1) else 2),
                *piece.interval).reshape(-1, 1)
            value = jnp.repeat(points, 2, axis=1)
        leaves.append(value)
    if prop == 'points':
        assert isinstance(actual, list)
        expected = [[leaves[k][:, j:j+1] for j in range(2)] for k in range(2)]
        if row:
            expected = [[expected[k][j] for k in range(2)] for j in range(2)]
        assert len(actual) == len(expected)
        for got_row, want_row in zip(actual, expected):
            for got, want in zip(got_row, want_row):
                np.testing.assert_array_equal(got, want)
    else:
        expected = jnp.concatenate(leaves, axis=0)
        if row:
            expected = expected.T
        assert not isinstance(actual, list)
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('prop', ['coeffs', 'values', 'points'])
@pytest.mark.parametrize('row', [False, True])
def test_source_one_row_multipiece_numericization(prop, row):
    f = cj.chebfun([lambda x: 2*jnp.ones_like(x), lambda x: 3*jnp.ones_like(x)], domain=(2., 3., 5.))
    obj = f.T if row else f
    leaves = get(obj, prop, 1)
    assert isinstance(leaves, list)
    actual = get(obj, prop, 2)
    expected = jnp.array([[2.5], [4.]]) if prop == 'points' else jnp.array([[2.], [3.]])
    if row:
        expected = expected.T
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_source_local_endpoints_actual_analytic_complex(f):
    left = get(f, 'lval-local')
    right = get(f, 'rval-local')
    ends = jnp.array([2., 3., 5.])
    def values(x):
        return jnp.stack((x+1j*x*x, 100*x+2-2j*x), axis=-1)
    expected_left, expected_right = values(ends[:-1]), values(ends[1:])
    scale = max(1., float(jnp.max(jnp.abs(expected_right))))
    assert float(jnp.max(jnp.abs(left-expected_left))) < 100*jnp.finfo(jnp.float64).eps*scale
    assert float(jnp.max(jnp.abs(right-expected_right))) < 100*jnp.finfo(jnp.float64).eps*scale


@pytest.mark.parametrize('row', [False, True])
@pytest.mark.parametrize('prop', ['lval-local', 'rval-local', 'vscale-local'])
def test_source_quasimatrix_one_row_numericization(row, prop):
    from chebfunjax.chebfun1d.linalg import Quasimatrix

    f = cj.chebfun(lambda x: jnp.stack((x, 100*x), axis=-1), domain=(2., 3., 5.))
    q = Quasimatrix([f.extract_columns(j) for j in range(2)], f.domain)
    actual = get(q.T if row else q, prop)
    expected = get(f.T if row else f, prop)
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('simplevel', [1, 2])
def test_source_ragged_row_outer_cell_transpose(simplevel):
    from chebfunjax.chebfun1d.linalg import Quasimatrix
    from chebfunjax.domain import Domain

    q = Quasimatrix([cj.chebfun(jnp.sin), cj.chebfun(jnp.cos, domain=(-1., 0., 1.))],
                    Domain((-1., 0., 1.)))
    column = get(q, 'coeffs', simplevel)
    row = get(q.T, 'coeffs', simplevel)
    assert len(column) == 2 and [len(c) for c in column] == [1, 2]
    assert len(row) == 2 and all(len(c) == 1 for c in row)
    for before, after in zip(column, row):
        assert len(before) == len(after[0])
        for actual, expected in zip(after[0], before):
            np.testing.assert_array_equal(actual, expected)
