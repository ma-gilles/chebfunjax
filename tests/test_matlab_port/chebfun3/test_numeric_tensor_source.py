"""Numeric ACA contracts and original constructor syntax pass(2).

Provenance
----------
MATLAB source: @chebfun3/chebfun3double.m;
tests/chebfun3/test_construnctorsyntax.m, Chebfun commit7574c77.
Analytic supplemental controls use the original constructor's 100eps scale;
they supplement, and do not replace, native tensor fixtures.
"""
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun3d._chebfun3double import complete_aca2, tensor_axes
from chebfunjax.chebfun3d.chebfun3 import Chebfun3, chebfun3
from chebfunjax.tech.chebtech import Chebtech2
from chebfunjax.tech.trigtech import Trigtech

EPS = np.finfo(float).eps


def test_native_diagonal_bias_strict_threshold_and_first_column_major_tie():
    matrix = jnp.asarray([[4., 5.], [5., 4.]])
    assert tuple(map(int, complete_aca2(matrix, 2.)[3][0])) == (0, 0)
    assert tuple(map(int, complete_aca2(matrix, 1.)[3][0])) == (1, 0)
    tied = jnp.asarray([[0., 8.], [8., 0.]])
    assert tuple(map(int, complete_aca2(tied, 0.)[3][0])) == (1, 0)
    assert bool(jnp.all(complete_aca2(jnp.ones((2, 3)), 1.)[1] == 0))


def test_complex_residual_skeleton_reconstruction_without_conjugation():
    values = jnp.asarray([[1.+1j, 2.+2j, 4.+4j], [2.+2j, 4.+4j, 8.+8j]])
    cols, pivots, rows, _ = complete_aca2(values, EPS)
    np.testing.assert_array_equal(cols @ jnp.diag(1/pivots) @ rows.T, values)


def test_original_constructor_syntax_pass2():
    domain = (-1., 1., -1., 1., 2., 5.)
    axes = tensor_axes((10, 15, 5), domain, Chebtech2)
    xx, yy, zz = jnp.meshgrid(*axes, indexing='ij')
    result = chebfun3(jnp.sin(xx*yy*zz), domain=domain)
    lengths = tuple(max(t.coeffs.shape[0] for t in group)
                    for group in (result.cols, result.rows, result.tubes))
    assert lengths == (10, 15, 5)


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('shape', [(1, 1, 1), (5,), (5, 4), (5, 4, 3)])
def test_numeric_constant_and_singleton_dimensions(periodic, shape):
    values = jnp.full(shape, 2.+1j)
    technology = Trigtech if periodic else Chebtech2
    result = Chebfun3.from_values(values, tech=technology)
    np.testing.assert_allclose(result(.23, -.17, .39), 2.+1j,
                               atol=100*EPS*abs(2.+1j), rtol=0)
    assert all(isinstance(t, technology) for group in
               (result.cols, result.rows, result.tubes) for t in group)


@pytest.mark.parametrize('periodic', [False, True])
def test_zero_numeric_tensor_compatibility(periodic):
    result = chebfun3(jnp.zeros((4, 3, 2)), trig=periodic)
    assert float(result(.23, -.17, .39)) == 0


def test_direct_periodic_complex_multimode_off_grid():
    domain = (1., 5., -3., 3., 2., 8.)
    axes = tensor_axes((8, 10, 12), domain, Trigtech)

    def exact(x, y, z):
        a = 2*jnp.pi*(x-1)/4
        b = 2*jnp.pi*(y+3)/6
        c = 2*jnp.pi*(z-2)/6
        return jnp.exp(1j*(a+2*b-c))+.25*jnp.cos(2*a)*jnp.sin(b)*jnp.cos(c)

    result = chebfun3(exact(*jnp.meshgrid(*axes, indexing='ij')), domain, trig=True)
    x = jnp.asarray([1.13, 2.71, 4.39])
    y = jnp.asarray([-2.17, .19, 2.31])
    z = jnp.asarray([2.23, 4.47, 7.11])
    expected = exact(x, y, z)
    assert float(jnp.max(jnp.abs(result(x, y, z)-expected))) < 100*EPS
    assert result.isPeriodicTech()


# These are additional native-capture controls, not original test predicates.
# The 100eps absolute convention is from tests/chebfun3/test_constructor.m:7.
# Original syntax pass(2) above retains its literal length equality.
_FIXTURE_DIR = Path(__file__).with_name('data')
_RECORDS = json.loads((_FIXTURE_DIR/'numeric_tensor_native.json').read_text())['records']


@pytest.mark.parametrize('record', _RECORDS, ids=[r['name'] for r in _RECORDS])
def test_native_numeric_tensor_capture(record, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Numeric source constructor must not use adaptive callbacks or HOSVD')

    monkeypatch.setattr(Chebfun3, 'from_function', classmethod(forbidden))
    monkeypatch.setattr(Chebfun3, 'from_coeffs', classmethod(forbidden))
    with np.load(_FIXTURE_DIR/'numeric_tensor_native.npz') as fixture:
        prefix = record['prefix']
        values = jnp.asarray(fixture[prefix+'_input'])
        result = chebfun3(values, tuple(record['domain']), trig=record['periodic'])
        groups = (result.cols, result.rows, result.tubes)
        assert list(result.core.shape) == record['rank']
        assert [max(t.coeffs.shape[0] for t in group) for group in groups] == record['lengths']
        assert [type(group[0]).__name__.lower() for group in groups] == record['tech']
        grid = [jnp.asarray(fixture[prefix+f'_axis{k}']) for k in range(3)]
        actual = result(*jnp.meshgrid(*grid, indexing='ij'))
        expected = jnp.asarray(fixture[prefix+'_grid_output'])
        assert float(jnp.max(jnp.abs(actual-expected))) < 100*EPS
        queries = [jnp.asarray(fixture[prefix+f'_query{k}']) for k in range(4)]
        assert float(jnp.max(jnp.abs(result(*queries[:3])-queries[3]))) < 100*EPS


@pytest.mark.parametrize('periodic', [False, True])
def test_nonzero_tensor_zero_iterations_at_coarse_source_epsilon(periodic):
    values = jnp.full((5, 4, 3), 2.75)
    result = chebfun3(values, (-2., 3., 4., 10., -7., -1.),
                      trig=periodic, eps=1.)
    assert result.rank == (0, 0, 0)
    assert result.length() == (1, 1, 1)
    assert result.core.shape == (1, 1, 1)
    np.testing.assert_array_equal(result.core, jnp.zeros((1, 1, 1)))
    groups = (result.cols, result.rows, result.tubes)
    assert [max(t.coeffs.shape[0] for t in group) for group in groups] == [5, 4, 3]
    assert float(result(.23, 5.17, -2.39)) == 0
    for group in groups:
        assert all(bool(jnp.all(t.coeffs == 0)) for t in group)
