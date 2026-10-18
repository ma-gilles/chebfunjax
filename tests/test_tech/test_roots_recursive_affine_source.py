"""Native recursive affine root maps at Chebfun7574c77.

Two immutable direct-component captures bind literal source stages. Original
inputs/storage are asserted in MATLAB before each source statement. Finite
words and nonfinite classes are strict; NaN sign/payload is not qualified.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.tech import chebtech as module

DATA = json.loads((Path(__file__).parents[1] / 'fixtures' /
                   'roots_recursive_affine_native_20261018.json').read_text())
ROWS = [row for capture in DATA['captures'] for row in capture['rows']]
PAIRS = [(rows[i], rows[i + 1]) for rows in
         [DATA['captures'][0]['rows'], DATA['captures'][1]['rows']]
         for i in range(0, len(rows), 2)]


def _array(record):
    if 0 in record['shape']:
        return np.empty(0, np.float64 if record['is_real'] else np.complex128)
    def component(name):
        return np.asarray([int(w, 16) for w in record[name]], np.uint64).view(np.float64)
    real, imag = component('re_hex'), component('im_hex')
    if record['is_real']:
        return real
    result = np.empty(real.shape, np.complex128)
    result.real, result.imag = real, imag
    return result


def _assert_components(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    for component in (np.real, np.imag):
        a, e = component(actual), component(expected)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(e))
        np.testing.assert_array_equal(np.isposinf(a), np.isposinf(e))
        np.testing.assert_array_equal(np.isneginf(a), np.isneginf(e))
        finite = np.isfinite(e)
        np.testing.assert_array_equal(a[finite].view(np.uint64), e[finite].view(np.uint64))


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['name'])
@pytest.mark.parametrize('mode', ['eager', 'jit'])
def test_literal_native_affine_component_words_and_classes(row, mode):
    roots = _array(row['input'])
    offset, scale = _array(row['offset'])[0], _array(row['scale'])[0]
    fn = module._roots_affine_jax if mode == 'jit' else module._roots_affine_jax.__wrapped__
    real_words, imag_words = fn(
        jnp.asarray(roots.real.view(np.uint64)),
        jnp.asarray(roots.imag.view(np.uint64)), offset, scale,
        complex_input=np.iscomplexobj(roots))
    assert isinstance(real_words, jax.Array) and isinstance(imag_words, jax.Array)
    expected = _array(row['literal'])
    actual = np.empty(expected.shape, expected.dtype)
    actual.real.view(np.uint64)[...] = np.asarray(real_words)
    if np.iscomplexobj(actual):
        actual.imag.view(np.uint64)[...] = np.asarray(imag_words)
    _assert_components(actual, expected)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda pair: pair[0]['name'])
def test_native_children_map_writable_storage_and_order(pair):
    left, right = [_array(r['input']) for r in pair]
    actual = module._roots_map_children(left, right)
    expected = np.concatenate([_array(r['literal']) for r in pair])
    assert isinstance(actual, np.ndarray) and actual.flags.writeable
    _assert_components(actual, expected)


@pytest.mark.parametrize('pair', PAIRS, ids=lambda pair: pair[0]['name'])
def test_actual_recursive_driver_uses_source_affine_map(pair, monkeypatch):
    left, right = [_array(r['input']) for r in pair]
    tokens = (np.array([1., 2., 3.]), np.array([4., 5., 6.]))
    original = module._roots_main
    seen = []
    def leaf(coefficients, *args, **kwargs):
        assert coefficients is tokens[0] or coefficients is tokens[1]
        seen.append(coefficients)
        return left if coefficients is tokens[0] else right
    monkeypatch.setattr(module, '_roots_subdivide', lambda _: tokens)
    monkeypatch.setattr(module, '_roots_main', leaf)
    c = np.zeros(51)
    c[-1] = 1.
    actual = original(c, 100 * np.finfo(float).eps, all_roots=True)
    assert len(seen) == 2
    _assert_components(actual, np.concatenate([_array(r['literal']) for r in pair]))


@pytest.mark.parametrize('li,ri,nl,nr', [
    (4, 1, 0, 75), (0, 5, 75, 0), (0, 3, 7, 129),
    (2, 1, 129, 7), (2, 3, 65, 1), (2, 3, 1, 65),
])
def test_native_asymmetric_children_and_storage_promotion(li, ri, nl, nr):
    left, right = _array(ROWS[li]['input'])[:nl], _array(ROWS[ri]['input'])[:nr]
    expected = np.concatenate([_array(ROWS[li]['literal'])[:nl],
                               _array(ROWS[ri]['literal'])[:nr]])
    actual = module._roots_map_children(left, right)
    assert actual.flags.writeable
    _assert_components(actual, expected)
