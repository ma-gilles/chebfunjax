"""Exact native completeACA controls for literal pivot-row division.

Provenance
----------
MATLAB source : @chebfun2/constructor.m (completeACA)
Chebfun commit: 7574c77680d7e82b79626300bf255498271a72df
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d._numeric_constructor import numeric_aca

_RECORDS = json.loads((Path(__file__).parents[1] / 'fixtures' /
                       'chebfun2_aca_updates_native_20261018.json').read_text())['cases']


def _decode(record):
    return np.asarray([int(v, 16) for v in record['re_hex']], dtype=np.uint64).view(
        np.float64).reshape(record['shape'], order='F')


@pytest.mark.parametrize('disabled', [False, True])
@pytest.mark.parametrize('record', _RECORDS, ids=['square17', 'refined33x51', 'lowbudget17'])
def test_native_complete_aca_division_words(record, disabled):
    with jax.disable_jit(disabled):
        pivots, positions, rows, cols, failed = numeric_aca(
            jnp.asarray(_decode(record['input'])),
            float(_decode(record['absolute']).item()), 4)
    assert failed == record['failed']
    np.testing.assert_array_equal(np.asarray(positions) + 1, _decode(record['positions']))
    for actual, key in [(pivots, 'pivots'), (rows, 'rows'), (cols, 'cols')]:
        expected = _decode(record[key])
        actual = np.asarray(actual).reshape(expected.shape)
        np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
