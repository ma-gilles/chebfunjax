"""Pinned constructor.m paired-axis, raw sample-test and EQUI controls.

The component bound is the existing coefficient-preference source control's
100*eps*max(1, norm(expected, inf)); shape, rank and classes are strict.
"""
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.chebfun2d.chebfun2 import chebfun2
from chebfunjax.chebpref import ChebfunPref

_DATA = json.loads((Path(__file__).parents[1] / 'fixtures' /
                    'chebfun2_paired_equi_source_20261018.json').read_text())
_CASES = {c['request']['name']: c for c in _DATA['cases']}
_GOOD = [name for name, c in _CASES.items() if c['ok']]


def _decode(r):
    if not np.prod(r['shape']):
        return np.empty(r['shape'])
    re = np.array([struct.unpack('>d', bytes.fromhex(w))[0] for w in r['re_hex']])
    im = np.array([struct.unpack('>d', bytes.fromhex(w))[0] for w in r['im_hex']])
    z = re if r.get('is_real', not r.get('complex_storage', False)) else re + 1j*im
    return z.reshape(r['shape'], order='F')


def _preference(r):
    if r is None:
        return None
    p = ChebfunPref(tech=r['tech'])
    p.cheb2Prefs.chebfun2eps = struct.unpack('>d', bytes.fromhex(r['eps_hex']))[0]
    p.cheb2Prefs.maxRank = r['max_rank']
    p.cheb2Prefs.sampleTest = r['sample_test']
    if r['override_min']:
        p.techPrefs.minSamples = r['override_min']
    if r['override_max']:
        p.techPrefs.maxLength = r['override_max']
    return p


def _operator(kind, calls):
    def op(x, y):
        calls.append(tuple(x.shape))
        if kind == 'trigx':
            z = jnp.cos(jnp.pi*x)
            return 1 + z/8 - y/4 + z*y/16
        if kind == 'trigy':
            z = jnp.cos(jnp.pi*y)
            return 1 + x/8 - z/4 + x*z/16
        if kind == 'rank7':
            v = jnp.ones_like(x+y)
            for k in range(1, 7):
                v = v + .25**k*x**k*y**k
            return v
        if kind == 'rank13':
            v = jnp.ones_like(x+y)
            tx0, tx, ty0, ty = jnp.ones_like(x), x, jnp.ones_like(y), y
            for k in range(1, 13):
                v = v + .25**k*tx*ty
                tx0, tx = tx, 2*x*tx-tx0
                ty0, ty = ty, 2*y*ty-ty0
            return v
        return 1 + x/8 - y/4 + x*y/16
    return op


def _construct(c, *, no_pref=False):
    r = c['request']
    calls = []
    pref = (_preference(r['xpref']) if r['single_pref'] else
            (_preference(r['xpref']),) if r.get('one_pref_cell') else
            (_preference(r['xpref']), _preference(r['ypref'])))
    kwargs = dict(domain=tuple(r['domain']))
    if not no_pref:
        kwargs['pref'] = pref
    kwargs.update({flag: True for flag in r['flags']})
    if r['sizes']:
        kwargs['n'] = tuple(r['sizes'])
    op = jnp.asarray(_decode(r['input'])) if r['kind'] == 'numeric' else _operator(r['op'], calls)
    return chebfun2(op, **kwargs), calls


@pytest.fixture(scope='module')
def constructed():
    return {name: _construct(_CASES[name]) for name in _GOOD}


def _close(actual, expected):
    actual, expected = np.asarray(actual), _decode(expected)
    assert actual.size == expected.size
    actual = actual.reshape(expected.shape)
    if actual.size:
        assert np.max(np.abs(actual-expected)) < 100*np.finfo(float).eps*max(1., np.max(np.abs(expected)))


@pytest.mark.parametrize('name', _GOOD)
def test_native_rank_classes_and_public_values(constructed, name):
    f, _ = constructed[name]
    a = _CASES[name]['actual']
    assert f.rank == a['rank']
    assert type(f.approx.cols[0]).__name__.lower() == a['col_class']
    assert type(f.approx.rows[0]).__name__.lower() == a['row_class']
    _close(f.approx.pivot_values, a['pivots'])
    _close(f(jnp.asarray(_decode(a['sample_x'])), jnp.asarray(_decode(a['sample_y']))), a['samples'])


# OPEN: remaining EQUI interpolant/final technology cutoff differences.
# Strict expected failures retain every component/shape assertion below.
_FACTOR_GAPS = {
    ('cols', 'equi_fixed_callable'): 'same native ACA factors: FUNQUI node/weight and final tech cutoff gap',
    ('rows', 'equix_fixed_callable'): 'same native factors and exact 129-query interpolant: final tech cutoff gap',
}


@pytest.mark.parametrize('axis,name', [
    pytest.param(axis, name, marks=pytest.mark.xfail(
        reason=_FACTOR_GAPS[axis, name], strict=True))
    if (axis, name) in _FACTOR_GAPS else pytest.param(axis, name)
    for axis in ['cols', 'rows'] for name in _GOOD
])
def test_native_factor_storage_and_components(constructed, name, axis):
    f, _ = constructed[name]
    actual = np.stack([t.coeffs for t in getattr(f.approx, axis)], axis=1)
    expected = _CASES[name]['actual'][axis]
    assert actual.shape == tuple(expected['shape'])
    _close(actual, expected)


@pytest.mark.parametrize('name', _GOOD)
def test_native_pivot_locations(constructed, name):
    f, _ = constructed[name]
    _close(np.asarray(f.pivot_locations), _CASES[name]['actual']['locations'])


@pytest.mark.parametrize('name', [n for n in _GOOD if _CASES[n]['request']['kind'] == 'adaptive'])
def test_native_scientific_callback_order(constructed, name):
    _, calls = constructed[name]
    expected = [tuple(c['shape']) for c in _CASES[name]['calls']]
    assert [c for c in calls if np.prod(c) > 4] == expected


@pytest.mark.parametrize('name,exception,identifier', [
    ('equi_callable_error', ValueError, 'CHEBFUN:CHEBFUN2:constructor:equi'),
    ('adaptive_one_pref_cell', IndexError, 'MATLAB:badsubscript'),
])
def test_source_error_order(name, exception, identifier):
    assert _CASES[name]['error_id'] == identifier
    with pytest.raises(exception, match=identifier):
        _construct(_CASES[name])


def test_default_equi_uses_source_numeric_route():
    c = _CASES['equi_shared']
    f, _ = _construct(c, no_pref=True)
    assert f.rank == c['actual']['rank']
    _close(np.stack([t.coeffs for t in f.approx.cols], axis=1), c['actual']['cols'])
    _close(np.stack([t.coeffs for t in f.approx.rows], axis=1), c['actual']['rows'])


def test_raw_sample_test_precedes_one_final_simplify(monkeypatch):
    from chebfunjax.chebfun2d import separable_approx as module
    events = []
    original = module._simplify_cheb_slices

    def simplify(*args, **kwargs):
        events.append('simplify')
        return original(*args, **kwargs)

    def op(x, y):
        events.append(('sample', tuple(x.shape)))
        return (1+x/8)*(1-y/4)

    monkeypatch.setattr(module, '_simplify_cheb_slices', simplify)
    raw, _, finalize, failure = module.SeparableApprox._construct_once(
        op, min_samples=(17, 17), techs=('cheb1', 'cheb'),
        _defer_simplify=True)
    assert not failure
    assert len(raw.cols[0].coeffs) == len(raw.rows[0].coeffs) == 17
    assert 'simplify' not in events
    before = list(events)
    final = finalize()
    assert len(final.cols[0].coeffs) == len(final.rows[0].coeffs) == 2
    assert events == before + ['simplify', 'simplify']
    with pytest.raises(RuntimeError, match='finalized twice'):
        finalize()
    events.clear()
    module.SeparableApprox.from_function(op, techs=('cheb1', 'cheb'))
    assert events == [('sample', (17, 17)), ('sample', (100, 1)),
                      'simplify', 'simplify']


@pytest.mark.parametrize('zero', [False, True])
def test_each_attempt_finalizes_once_without_resampling(zero):
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    calls = []

    def op(x, y):
        calls.append(tuple(x.shape))
        return 0*x+0*y if zero else (1+x/8)*(1-y/4)

    _, _, finalize, _ = SeparableApprox._construct_once(op, _defer_simplify=True)
    before = list(calls)
    finalize()
    assert calls == before
    with pytest.raises(RuntimeError, match='finalized twice'):
        finalize()
    assert calls == before


def test_failed_attempt_stops_after_source_sample_test(monkeypatch):
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    attempts, samples, finalizations = [], [], []

    class Raw:
        def __call__(self, x, y):
            return jnp.zeros_like(x+y)

    def once(cls, op, **kwargs):
        attempts.append(kwargs)
        def finalize():
            finalizations.append(True)
            return 'final'
        return Raw(), np.finfo(float).eps, finalize, True

    def op(x, y):
        samples.append(x.shape)
        return jnp.ones_like(x+y)

    monkeypatch.setattr(SeparableApprox, '_construct_once', classmethod(once))
    assert SeparableApprox.from_function(op) == 'final'
    assert len(attempts) == len(finalizations) == 1
    assert samples == [(100, 1)]


@pytest.mark.parametrize('axis', [0, 1])
def test_axis_sample_budget_checked_after_skeleton_update(axis):
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    calls = []

    def op(x, y):
        calls.append((tuple(x.shape), np.asarray(x), np.asarray(y)))
        return jnp.sin(50*x)*(1+y/4) if axis == 0 else (1+x/4)*jnp.sin(50*y)

    techs = ('cheb1', 'cheb') if axis == 0 else ('cheb', 'cheb1')
    limits = (17, 65537) if axis == 0 else (65537, 17)
    with pytest.warns(RuntimeWarning, match='maximum length'):
        raw, _, _, failed = SeparableApprox._construct_once(
            op, techs=techs, max_samples=limits, _defer_simplify=True)
    assert failed
    # Literal source order: resample unresolved C1 skeleton to 3*17,
    # update its happiness, then compare both resulting dimensions >= limits.
    assert len(raw.rows[0].coeffs if axis == 0 else raw.cols[0].coeffs) == 51
    # Existing Python skeleton callbacks use flat vectors. Native source
    # uses row/column arrays; this asserts requested content/length only.
    records = [(x, y) for shape, x, y in calls if shape == (51,)]
    assert records
    x, y = records[0]
    varying, constant = (x, y) if axis == 0 else (y, x)
    assert np.ptp(varying) > 1
    assert np.ptp(constant) == 0
    assert np.isfinite(varying).all() and np.isfinite(constant).all()


@pytest.mark.parametrize('index,techs', [
    (0, ('chebtech1', 'chebtech2')),
    (1, ('trigtech', 'chebtech2')),
    (2, ('chebtech1', 'trigtech')),
])
def test_native_adaptive_zero_axis_classes_and_metadata(index, techs):
    f = chebfun2(lambda x, y: 0*x+0*y,
                 pref=tuple(ChebfunPref(tech=t) for t in techs))
    a = _DATA['adaptive_zeros'][index]
    assert f.rank == 1
    assert type(f.approx.cols[0]).__name__.lower() == a['col_class']
    assert type(f.approx.rows[0]).__name__.lower() == a['row_class']
    _close(f.approx.cols[0].coeffs, a['cols'])
    _close(f.approx.rows[0].coeffs, a['rows'])
    assert np.isposinf(np.asarray(f.approx.pivot_values)).all()
    _close(f.pivot_locations, a['locations'])
    assert float(f(jnp.asarray(.3), jnp.asarray(-.7))) == 0


@pytest.mark.parametrize('finite_miss,expected_attempts', [(False, 1), (True, 2)])
def test_source_sample_decision_omits_nan_comparisons(monkeypatch, finite_miss, expected_attempts):
    assert _DATA['sample_decisions'][int(finite_miss)]['pass'] is (not finite_miss)
    from chebfunjax.chebfun2d.separable_approx import SeparableApprox
    attempts, finalizations = [], []

    class Raw:
        def __call__(self, x, y):
            return jnp.zeros_like(x+y)

    def once(cls, op, **kwargs):
        attempts.append(kwargs)
        assert len(attempts) <= expected_attempts
        def finalize():
            finalizations.append(True)
            return 'final'
        return Raw(), np.finfo(float).eps, finalize, len(attempts) == 2

    def op(x, y):
        result = jnp.full_like(x+y, jnp.nan)
        return result.at[0].set(1) if finite_miss else result

    monkeypatch.setattr(SeparableApprox, '_construct_once', classmethod(once))
    # Native any(max(abs(error)) > threshold): max omits NaNs. All NaNs
    # compare false; a finite miss among them rejects the first attempt.
    assert SeparableApprox.from_function(op) == 'final'
    assert len(attempts) == expected_attempts
    assert len(finalizations) == 1
