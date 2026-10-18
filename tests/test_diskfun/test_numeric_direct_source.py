"""Native constructFromDouble returns projected full matrix factors, no simplify."""
import importlib
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.diskfun.diskfun import Diskfun

RAW = json.loads((Path(__file__).parents[1]/'fixtures/disk_numeric_direct_native_20261018.json').read_text())

def array(record):
    return np.asarray([struct.unpack('>d', bytes.fromhex(w))[0] for w in record['re_hex']]).reshape(record['shape'], order='F')

def words(actual, expected):
    actual = np.asarray(actual)
    assert list(actual.shape) == expected['shape']
    for component, key in [(actual.real, 're_hex'), (actual.imag, 'im_hex')]:
        native = np.asarray([int(w, 16) for w in expected[key]], np.uint64).reshape(expected['shape'], order='F')
        np.testing.assert_array_equal(np.ascontiguousarray(component).view(np.uint64), native)

@pytest.mark.parametrize('case', RAW['cases'], ids=lambda c:c['name'])
def test_default_numeric_native_storage_and_metadata(case):
    f = Diskfun.from_values(jnp.asarray(array(case['input'])))
    assert [len(f.cols[0].coeffs), len(f)] == case['cols']['shape']
    assert {len(t.coeffs) for t in f.cols} == {case['cols']['shape'][0]}
    assert {len(t.coeffs) for t in f.rows} == {case['rows']['shape'][0]}
    words(jnp.asarray(f.pivot_locations), case['locations'])
    words(f.pivots.reshape(-1, 1), case['pivots'])
    assert list(f.idx_plus) == (array(case['plus']).reshape(-1)-1).tolist()
    assert list(f.idx_minus) == (array(case['minus']).reshape(-1)-1).tolist()
    assert f.nonzero_poles == case['pole']

@pytest.mark.parametrize('case', [c for c in RAW['cases'] if c['name'] != '3x8'], ids=lambda c:c['name'])
def test_default_numeric_native_row_words(case):
    f = Diskfun.from_values(jnp.asarray(array(case['input'])))
    words(jnp.stack([t.coeffs for t in f.rows], axis=1), case['rows'])

def test_default_numeric_native_all_factor_words_2x4():
    case = RAW['cases'][0]
    f = Diskfun.from_values(jnp.asarray(array(case['input'])))
    words(jnp.stack([t.coeffs for t in f.cols], axis=1), case['cols'])
    words(jnp.stack([t.coeffs for t in f.rows], axis=1), case['rows'])

@pytest.mark.parametrize('option', [None, False, True])
def test_all_numeric_entry_options_bypass_legacy_host_simplification(option, monkeypatch):
    module = importlib.import_module('chebfunjax.diskfun.diskfun')
    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError('host NumPy accessed: '+name)
    monkeypatch.setattr(module, 'np', ForbidNumpy())
    def forbidden(*args, **kwargs):
        raise AssertionError('native numeric constructor never simplifies or prunes')
    monkeypatch.setattr(Diskfun, 'simplify', forbidden)
    monkeypatch.setattr(Diskfun, '_prune_zero_terms', forbidden)
    values=jnp.asarray([[1.,1.,1.,1.],[2.,2.,2.,2.]])
    kwargs={} if option is None else {'preserve_lengths':option}
    f=Diskfun.from_values(values, **kwargs)
    assert len(f)==1

GE = json.loads((Path(__file__).parents[1]/'fixtures/disk_phaseone_jax_native_20261018.json').read_text())
ERRORS = {'one_inf3x4', 'all_nan2x4', 'extreme2x4', 'mean_cancel2x8'}

@pytest.mark.parametrize('case', GE['cases']+GE['public_tiny_cases'], ids=lambda c:c['name'])
@pytest.mark.parametrize('option', [None, False])
def test_default_and_false_native_controls_without_host_numpy(case, option, monkeypatch):
    values = jnp.asarray(array(case['input']))
    class ForbidNumpy:
        def __getattr__(self, name):
            raise AssertionError('host NumPy accessed: '+name)
    for name in ['chebfunjax.diskfun.diskfun', 'chebfunjax.tech.chebtech', 'chebfunjax.tech.trigtech']:
        monkeypatch.setattr(importlib.import_module(name), 'np', ForbidNumpy())
    kwargs = {} if option is None else {'preserve_lengths':option}
    if case['name'] in ERRORS:
        with pytest.raises(IndexError, match='MATLAB:badsubscript: Index in position 2 exceeds array bounds.'):
            Diskfun.from_values(values, **kwargs)
    else:
        f = Diskfun.from_values(values, **kwargs)
        assert len(f) > 0
        assert len(f.cols) == len(f.rows) == len(f.pivots)

CAP = json.loads((Path(__file__).parents[1]/'fixtures/disk_numeric_cap_native_20261018.json').read_text())

@pytest.mark.parametrize('case', CAP['cases'], ids=lambda c:c['name'])
def test_literal_native_rank_cap_storage_pivots_classes_and_values(case):
    f = Diskfun.from_values(jnp.asarray(array(case['input'])))
    assert len(f) == case['rank']
    words(f.pivots.reshape(-1,1), case['pivots'])
    words(jnp.asarray(f.pivot_locations), case['locations'])
    for key, techs in [('cols', f.cols), ('rows', f.rows)]:
        coefficients = np.asarray(jnp.stack([t.coeffs for t in techs], axis=1))
        assert list(coefficients.shape) == case[key]['shape']
        for component, native_key in [(coefficients.real, 're_hex'), (coefficients.imag, 'im_hex')]:
            native = np.asarray([struct.unpack('>d', bytes.fromhex(w))[0] for w in case[key][native_key]]).reshape(case[key]['shape'], order='F')
            np.testing.assert_array_equal(np.isnan(component), np.isnan(native))
            np.testing.assert_array_equal(np.isposinf(component), np.isposinf(native))
            np.testing.assert_array_equal(np.isneginf(component), np.isneginf(native))
    for tkey, rkey, vkey in [('theta', 'radius', 'values'), ('off_theta', 'off_radius', 'off_values')]:
        actual = f.fevalm(jnp.asarray(array(case[tkey])).reshape(-1), jnp.asarray(array(case[rkey])).reshape(-1))
        # Reuse the unchanged authored1e-12 accuracy criterion against the
        # actual pinned native result. The analytic interpolation demand is
        # separately retained, strictly expected to fail for both providers.
        assert float(jnp.max(jnp.abs(actual-jnp.asarray(array(case[vkey]))))) < 1e-12
