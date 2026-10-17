"""Retained R2025b builtin numeric indexing and literal Chebfun recursion.

These65 API captures are new source controls, not original MATLAB clauses.
Native constructor roundoff is not imposed on the Python constructor: public
recursion checks actual computed data using independently bound source indices.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

_REF = json.loads((Path(__file__).parent/'references/subsref_numeric_recursion_native_20261017.json').read_text())
_CASES = {r['label']: r for r in _REF['records']}
_PRIMITIVES = [r['label'] for r in _REF['records'] if r['number'] <= 53]
_PUBLIC = [r['label'] for r in _REF['records'] if r['number'] > 53]


def _value(record):
    kind = record['kind']
    if kind == 'char':
        return record['text']
    if kind == 'opaque':
        # MATLAB string storage has no Python class adapter: an explicit
        # Unicode ndarray models this nonnumeric-index type, not a char.
        return np.array('1')
    if kind == 'cell':
        out = np.empty(len(record['data']), dtype=object)
        out[:] = [_value(v) for v in record['data']]
        return out.reshape(record['shape'], order='F')
    values = np.asarray(record['real'], dtype=record['dtype'])
    if record['imag'] is not None:
        values = values + 1j*np.asarray(record['imag'])
    return jnp.asarray(values.reshape(record['shape'], order='F'))


def _records(reference):
    return [{'type': r['type']['text'],
             'subs': [_value(v) for v in r['subs']['data']] if r['subs']['kind'] == 'cell'
                     else r['subs']['text']}
            for r in reference['indices']['data']]


@pytest.mark.parametrize('label', _PRIMITIVES)
def test_actual_native_builtin(label):
    from chebfunjax.chebfun1d._numeric_subsref import source_numeric_subsref
    ref = _CASES[label]
    value, records = _value(ref['input']), _records(ref)
    if not ref['success']:
        with pytest.raises(ValueError) as caught:
            source_numeric_subsref(value, records)
        assert caught.value.identifier == ref['identifier']
        return
    actual = source_numeric_subsref(value, records)
    expected = _value(ref['output'])
    assert actual.shape == tuple(ref['output_size'])
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.fixture(scope='module')
def f():
    return cj.chebfun(lambda x: jnp.stack((x, x*x), axis=-1), domain=[-2., 3.])


def _first(label):
    if label.startswith(('feval_', 'three_record')):
        x = jnp.linspace(-.75, .75, 8).reshape((2, 2, 2), order='F') if label == 'feval_nd_fold' else jnp.array([[0.], [.5]])
        return {'type': '()', 'subs': [x]}
    if label == 'row_feval':
        return {'type': '()', 'subs': [jnp.array([[0.], [.25], [.5]])]}
    prop = {'domain_linear': 'domain', 'ends_logical': 'ends',
            'domain_column_index': 'domain', 'lval_linear': 'lval',
            'rval_permutation': 'rval', 'vscale_scalar': 'vscale'}[label]
    return {'type': '.', 'subs': prop}


# Independent source positions, verified against actual retained native input
# and output arrays below. They do not supply constructor/evaluation values.
_MAP = {'feval_linear': [1], 'feval_column': [3, 0],
        'feval_cartesian': [3, 2, 1, 0], 'feval_nd_fold': list(range(16)),
        'row_feval': [4, 5, 0, 1], 'domain_linear': [1],
        'ends_logical': [0], 'domain_column_index': [1, 0],
        'lval_linear': [1], 'rval_permutation': [1, 0], 'vscale_scalar': [0]}


@pytest.mark.parametrize('label', _PUBLIC)
def test_actual_native_public_recursion(f, label):
    ref = _CASES[label]
    receiver = f.T if label == 'row_feval' else f
    first = _first(label)
    records = [first] + _records(ref)
    if not ref['success']:
        with pytest.raises(ValueError) as caught:
            receiver.subsref(records)
        assert caught.value.identifier == ref['identifier']
        return
    positions = _MAP[label]
    # First bind the independently specified index map to the actual native
    # captured data; then apply it to the actual Python computed data.
    native_input = np.asarray(_value(ref['input'])).reshape(-1, order='F')
    native_output = np.asarray(_value(ref['output'])).reshape(-1, order='F')
    np.testing.assert_array_equal(native_input[positions], native_output)
    computed = jnp.asarray(receiver.subsref(first)).reshape(-1, order='F')
    expected = computed[jnp.asarray(positions)].reshape(ref['output_size'], order='F')
    actual = receiver.subsref(records)
    assert actual.shape == tuple(ref['output_size'])
    np.testing.assert_array_equal(actual, expected)


def test_numeric_recursion_jit_ad_preserves_complex_values(f):
    records = [{'type': '()', 'subs': [jnp.array([[.25], [.5]])]},
               {'type': '()', 'subs': [[2., 1.], [2., 1.]]}]
    def evaluate(x):
        return f.subsref([{'type': '()', 'subs': [x]}, records[1]])
    def expected(x):
        full = cj.feval(f, x)
        return full[jnp.ix_(jnp.array([1, 0]), jnp.array([1, 0]))]
    x = jnp.array([[.25], [.5]])
    np.testing.assert_array_equal(jax.jit(evaluate)(x), jax.jit(expected)(x))
    np.testing.assert_array_equal(jax.jacfwd(evaluate)(x), jax.jacfwd(expected)(x))
    z = x + .125j
    actual = jax.jit(evaluate)(z)
    assert jnp.iscomplexobj(actual)
    np.testing.assert_array_equal(actual, jax.jit(expected)(z))


def test_numeric_get_domain_records_preserve_public_tuple(f):
    assert isinstance(f.get('domain'), tuple)
    actual = f.subsref([{'type': '.', 'subs': 'domain'}, {'type': '()', 'subs': [':']}])
    np.testing.assert_array_equal(actual, jnp.array([[-2.], [3.]]))
    assert isinstance(f.get('domain'), tuple)


def test_cell_get_result_stays_explicitly_unsupported(f):
    assert isinstance(f.get('funs'), list)
    with pytest.raises(NotImplementedError):
        f.subsref([{'type': '.', 'subs': 'funs'}, {'type': '{}', 'subs': [1]}])


def test_source_numeric_recursion_actual_polynomial(f):
    x = jnp.array([[-.75], [.25], [.5]])
    actual = f.subsref([{'type': '()', 'subs': [x]},
                        {'type': '()', 'subs': [':', [2, 1]]}])
    expected = jnp.concatenate((x*x, x), axis=1)
    assert float(jnp.max(jnp.abs(actual-expected))) < 100*jnp.finfo(jnp.float64).eps
