"""Actual native record-index probe controls, not original MATLAB clauses.

Reference: source7574c77,48 actual default polynomial Chebfun calls captured
under R2025b. Index expectations are derived from actual native output columns;
constructor coefficients/degrees are never supplied to the candidate.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import chebfunjax as cj

_REFERENCE = json.loads((Path(__file__).parent/'references/subsref_record_indices_native_20261017.json').read_text())
_CASES = {(c['label'], c['mode']): c for c in _REFERENCE['cases']}
_SELECTORS = {
    'one': 1., 'reorder_repeat': [2., 1., 2.], 'column_vector': [[2.], [1.]],
    'empty': [], 'fractional': 1.5, 'zero': 0., 'negative': -1.,
    'mixed_fractional': [1., 1.5], 'logical_true': True, 'logical_false': False,
    'logical_mask': [True, False], 'logical_true_true': [True, True],
    'complex_nonreal': complex(1., 1.), 'complex_zero_imag': complex(1., 0.),
    'mixed_complex': [complex(1., 1.), 1.], 'nan': float('nan'),
    'mixed_nan': [float('nan'), 1.], 'positive_inf': float('inf'),
    'negative_inf': -float('inf'), 'matrix_inside': [[1., 2.], [2., 1.]],
    'matrix_outside': [[3., 3.], [3., 3.]],
    'matrix_mixed': [[1., 3.], [1., 3.]], 'unknown_char': 'bogus', 'alias_left': 'left',
}


@pytest.fixture(scope='module')
def f():
    return cj.chebfun(lambda x: jnp.stack((x, x**2), axis=-1))


@pytest.mark.parametrize('label', list(_SELECTORS))
@pytest.mark.parametrize('mode', ['numeric', 'colon'])
def test_actual_native_probe(f, label, mode):
    reference = _CASES[label, mode]
    x = jnp.asarray([[.25], [.5]]) if mode == 'numeric' else ':'
    record = {'type': '()', 'subs': [x, _SELECTORS[label]]}
    if not reference['success']:
        with pytest.raises(Exception) as caught:
            f.subsref(record)
        assert getattr(caught.value, 'identifier', None) == reference['identifier']
        return
    actual = f.subsref(record)
    columns = reference['selected_columns']
    if mode == 'numeric':
        expected = cj.feval(f, x)[:, columns]
        assert actual.shape == tuple(reference['output_shape'])
        np.testing.assert_array_equal(actual, expected)
    elif columns:
        assert actual.isequal(f[:, columns])
    else:
        assert actual.isempty()
        assert actual.domain == f.domain


@pytest.mark.parametrize(('shape', 'column', 'blocks', 'error'), [
    ((), [2., 1.], [1, 0], None),
    ((2, 1), [2., 1.], [1, 0], None),
    ((1, 2), 1.5, [1, 2], None),
    ((2, 2), 1.5, [1, 2], None),
    ((2, 2, 2), [2., 1., 2.], [2, 3, 0, 1, 2, 3], None),
    ((2, 0), 1.5, [], None),
    ((0, 0), 0., [], None),
    ((0, 2), 1., [0, 1], None),
    ((0, 2), 2., None, 'MATLAB:badsubscript'),
    ((2, 1, 0), 1.5, None, 'MATLAB:badsubscript'),
])
def test_source_computed_outcols_shapes(f, shape, column, blocks, error):
    # Source arithmetic validates computed outCols, not raw component numbers.
    # Empty FEVAL preserves query shape and skips component packing.
    x = jnp.asarray(.5) if not shape else jnp.zeros(shape)
    if error:
        with pytest.raises(ValueError) as caught:
            f.subsref({'type': '()', 'subs': [x, column]})
        assert caught.value.identifier == error
        return
    actual = f.subsref({'type': '()', 'subs': [x, column]})
    full = cj.feval(f, x.reshape((1, 1)) if not shape else x)
    expected = full[:, blocks, ...]
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(('column', 'expected_columns'), [
    ([2., 1.], [1, 0]), (complex(1., 0.), [0]),
    (float('nan'), [0, 1]), ([True, True], [0, 0]),
])
def test_static_record_selection_jit_ad(f, column, expected_columns):
    def evaluate(x):
        return f.subsref({'type': '()', 'subs': [x, column]})
    def expected(x):
        return f(x, columns=expected_columns)
    x = jnp.asarray([[.25], [.5]])
    actual = jax.jit(evaluate)(x)
    assert isinstance(actual, jax.Array)
    np.testing.assert_array_equal(actual, expected(x))
    np.testing.assert_array_equal(jax.jacfwd(evaluate)(x), jax.jacfwd(expected)(x))
