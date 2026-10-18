"""Native coefficient/numeric constructor storage, Chebfun7574c77.

Original fixtures and bounds are unchanged. New factor comparisons use a
predeclared rounding allowance; exact origin-only source fields stay strict.
"""
import json
import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun, _phase_one_matrix_disk

FIXTURE = Path(__file__).parents[1]/'fixtures'/'disk_coeffs_numeric_storage_native_20261018.json'
CASES = json.loads(FIXTURE.read_text())['cases']


def _parts(record):
    shape = tuple(record['shape'])
    return tuple(np.asarray([struct.unpack('>d', bytes.fromhex(w))[0] for w in record[k]])
                 .reshape(shape, order='F') for k in ['re_hex', 'im_hex'])


def _array(record):
    re, im = _parts(record)
    return jnp.asarray(re) if record['is_real'] else jax.lax.complex(jnp.asarray(re), jnp.asarray(im))


def _assert_words(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64))


def _check(g, native, *, strict=False):
    assert len(g) == native['rank']
    assert [t.n for t in g.cols] == [native['cols']['shape'][0]]*len(g)
    assert [t.n for t in g.rows] == [native['rows']['shape'][0]]*len(g)
    assert list(g.idx_plus) == [int(i)-1 for i in np.atleast_1d(native['idx_plus'])]
    assert list(g.idx_minus) == [int(i)-1 for i in np.atleast_1d(native['idx_minus'])]
    assert g.nonzero_poles == native['nonzero_poles']
    for t, flag in zip(g.rows, np.atleast_1d(native['row_is_real'])):
        assert t.real_columns == (bool(flag),)
    _assert_words(np.asarray(g.pivots), _parts(native['pivots'])[0].reshape(-1))
    _assert_words(np.asarray(g.pivot_locations), _parts(native['locations'])[0])
    for kind, techs in [('cols', g.cols), ('rows', g.rows)]:
        re, im = _parts(native[kind])
        for j, tech in enumerate(techs):
            actual = np.asarray(tech.coeffs)
            if strict:
                _assert_words(actual.real, re[:, j])
                _assert_words(actual.imag.copy(), im[:, j])
            else:
                expected = re[:, j]+1j*im[:, j]
                # New allowance, not an original native tolerance: both FFT
                # transforms and GE factors contain O(n) scalar accumulations.
                #64*n eps times total factor scale covers their operation count
                # in these bounded inputs; no observed delta determines it.
                scale = np.sum(np.abs(actual))+np.sum(np.abs(expected))
                bound = 64*actual.size*np.finfo(float).eps*scale
                assert np.max(np.abs(actual-expected), initial=0.) <= bound


@pytest.mark.parametrize('index', range(len(CASES)))
def test_coefficient_route_retains_native_numeric_storage(index):
    case = CASES[index]
    _check(Diskfun.coeffs2diskfun(_array(case['input'])), case['actual'])


@pytest.mark.parametrize('index', range(len(CASES)))
def test_native_numeric_input_isolates_constructor_from_fft_prefix(index):
    case = CASES[index]
    _check(Diskfun.from_values(_array(case['numeric_input']), preserve_lengths=True),
           case['literal_numeric'])


@pytest.mark.parametrize('index', [3, 4, 5, 6])
def test_native_origin_only_and_general_zero_component_words(index):
    case = CASES[index]
    _check(Diskfun.coeffs2diskfun(_array(case['input'])), case['actual'], strict=True)


@pytest.mark.parametrize('values', [[[2., -3., 4., -5.]], [[-2., 3.]]])
def test_origin_only_source_precedes_empty_gradient_tolerance(values):
    # Literal native PhaseOne m==1: only first origin datum is retained;
    # angular row is ones, pivot1. No tolerance or gradient is consumed.
    g = Diskfun.from_values(jnp.asarray(values), preserve_lengths=True)
    _assert_words(g.cols[0].coeffs, np.asarray([values[0][0]]))
    _assert_words(g.rows[0].values, np.ones(len(values[0])))
    _assert_words(g.pivots, np.ones(1))
    assert g.idx_plus == (0,) and g.idx_minus == () and g.nonzero_poles


def test_phase_one_origin_only_literal_pivot():
    result = _phase_one_matrix_disk(np.asarray([[2., -3., 4., -5.]]), None, 100.)
    _assert_words(result[4], np.ones(1))
    _assert_words(result[5], np.ones((4, 1)))
    assert result[2] and result[6] == [0] and result[7] == []


def test_real_projection_still_enters_literal_numeric_constructor(monkeypatch):
    captured = []
    marker = object()
    def collect(cls, values, **kwargs):
        captured.append((values, kwargs))
        return marker
    monkeypatch.setattr(Diskfun, 'from_values', classmethod(collect))
    assert Diskfun.coeffs2diskfun(jnp.asarray([[1.+2j]])) is marker
    _assert_words(captured[0][0], np.ones((1, 2)))
    assert captured[0][1] == {'preserve_lengths': True}


@pytest.mark.parametrize('index', range(len(CASES)))
def test_independent_polynomial_values_after_source_storage(index):
    case = CASES[index]
    g = Diskfun.coeffs2diskfun(_array(case['input']))
    theta = jnp.asarray([-jnp.pi, -.7, 0., .4, jnp.pi])
    radius = jnp.asarray([0., .125, .5, .8, 1.])
    name = case['name']
    if 'profile' in name:
        expected = .25*(radius**2-1)
    elif name == 'odd_even_cubic4x7':
        expected = radius**3*jnp.cos(3*theta)+radius**2*jnp.sin(theta)**2
    elif name == 'complex_scalar1x1':
        expected = jnp.ones_like(radius)
    elif name == 'mixed_parity9x8':
        expected = .5+.25*radius*jnp.cos(theta)
    elif name == 'r_sin2x3':
        expected = radius*jnp.sin(theta)
    else:
        expected = jnp.zeros_like(radius)
    # Same10eps functional bound as the unchanged original coefficient module.
    assert float(jnp.max(jnp.abs(g(theta, radius)-expected))) < 10*jnp.finfo(jnp.float64).eps
