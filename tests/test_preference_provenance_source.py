"""Raw preference source contracts: @chebfunpref/chebfunpref.m, pin7574c77.

Python mapping mutation/copy extensions are labelled separately. No numerics.
"""
import copy

import pytest

from chebfunjax.chebpref import ChebfunPref, ChebopPref, DotDict


@pytest.fixture(autouse=True)
def isolated_sessions():
    saved = ChebfunPref._defaults, ChebopPref._defaults
    ChebfunPref.setDefaults('factory')
    ChebopPref.setDefaults('factory')
    yield
    ChebfunPref._defaults, ChebopPref._defaults = saved


@pytest.mark.parametrize('tech,maximum', [
    ('chebtech', 65537), ('chebtech1', 65537), ('chebtech2', 65537),
    ('trigtech', 65536), ('trig', 65536), ('periodic', 65536),
    ('@TRIGTECH', 65536),
])
def test_selected_defaults_without_raw_materialization(tech, maximum):
    p = ChebfunPref(tech=tech)
    assert p.maxLength == p.techPrefs.maxLength == maximum
    assert p._tech_overrides == {}
    assert p.minSamples == 17 and p.fixedLength is None
    assert ('gridType' in p.techPrefs) is (maximum == 65536)
    assert ('useTurbo' in p.techPrefs) is (maximum == 65537)


@pytest.mark.parametrize('route', ['dict', 'kwargs', 'attribute', 'nested', 'item', 'update', 'union'])
def test_explicit_equal_polynomial_default_survives_trig(route):
    p = ChebfunPref()
    if route == 'dict':
        p = ChebfunPref({'maxLength': 65537})
    elif route == 'kwargs':
        p = ChebfunPref(maxLength=65537)
    elif route == 'attribute':
        p.maxLength = 65537
    elif route == 'nested':
        p.techPrefs.maxLength = 65537
    elif route == 'item':
        p.techPrefs['maxLength'] = 65537
    elif route == 'update':
        p.techPrefs.update(maxLength=65537)
    else:
        p.techPrefs |= {'maxLength': 65537}
    p.tech = 'trigtech'
    assert p.maxLength == 65537
    assert p._tech_overrides == {'maxLength': 65537}


def test_retained_view_refresh_and_setdefault_read():
    p = ChebfunPref()
    view = p.techPrefs
    assert view.setdefault('maxLength', 42) == 65537
    assert p._tech_overrides == {}
    p.tech = 'trigtech'
    assert view is p.techPrefs and view.maxLength == 65536
    view.setdefault('custom', {'child': 3})
    view.custom.child = 4
    assert p._tech_overrides['custom']['child'] == 4


def test_session_partial_dict_and_keywords_source():
    ChebfunPref.setDefaults(domain=(-2, 3), maxLength=123, sampleTest=False)
    for p in (ChebfunPref({'minSamples': 21}), ChebfunPref(minSamples=21)):
        assert p.domain == (-2, 3) and p.maxLength == 123
        assert p.sampleTest is False and p.minSamples == 21
    assert ChebfunPref(maxLength=45).maxLength == 45


@pytest.mark.parametrize('copier', [ChebfunPref, copy.copy, copy.deepcopy])
def test_object_copy_preserves_raw_and_isolates_nested(copier):
    p = ChebfunPref(custom={'child': 3})
    p.techPrefs
    q = copier(p)
    q.tech = 'trigtech'
    q.custom.child = 4
    assert q.maxLength == 65536 and p.maxLength == 65537
    assert p.custom.child == 3
    assert 'maxLength' not in q._tech_overrides


def test_object_input_ignores_later_session():
    p = ChebfunPref()
    ChebfunPref.setDefaults(maxLength=99)
    assert ChebfunPref(p).maxLength == 65537
    ChebfunPref.setDefaults(p)
    q = ChebfunPref(tech='trigtech')
    assert q.maxLength == 65536


@pytest.mark.parametrize('copier', [dict, copy.copy, copy.deepcopy, lambda x: x.copy()])
def test_public_view_materialization_is_explicit(copier):
    p = ChebfunPref()
    supplied = copier(p.techPrefs)
    assert isinstance(supplied, dict)
    q = ChebfunPref({'techPrefs': supplied, 'tech': 'trigtech'})
    assert q.maxLength == 65537
    supplied['maxLength'] = 9
    assert p.maxLength == 65537 and q.maxLength == 65537


def test_two_input_struct_partial_vs_object_materialized_source():
    base = ChebfunPref(maxLength=91, sampleTest=False)
    partial = ChebfunPref(base, {'tech': 'trigtech'})
    assert partial.maxLength == 91 and partial.sampleTest is False
    other = ChebfunPref(tech='trigtech')
    full = ChebfunPref(base, other)
    assert full.maxLength == 65536 and full.sampleTest is True
    full.tech = 'chebtech2'
    assert full.maxLength == 65536
    assert base.maxLength == 91 and other._tech_overrides == {}


def test_constructor_shallow_substructure_merge_source():
    p = ChebfunPref({'splitPrefs': {'splitLength': 42}})
    assert p.splitPrefs.splitMaxLength == 6000
    q = ChebfunPref(p, {'splitPrefs': {'custom': {'a': 1, 'b': 2}}})
    r = ChebfunPref(q, {'splitPrefs': {'custom': {'a': 3}}})
    assert r.splitPrefs.custom == {'a': 3}
    r.splitPrefs = {'splitLength': 5}
    assert 'splitMaxLength' not in r.splitPrefs


def test_merge_materializes_own_tech_and_later_wins_source():
    poly = ChebfunPref()
    trig = ChebfunPref(tech='trigtech')
    merged = ChebfunPref.mergeTechPrefs(poly, trig)
    assert merged.maxLength == 65536
    assert merged.gridType == 2 and merged.useTurbo is False
    assert ChebfunPref.mergeTechPrefs(trig, poly).maxLength == 65537
    assert ChebfunPref.mergeTechPrefs(trig, {'maxLength': 7}).maxLength == 7
    assert poly._tech_overrides == trig._tech_overrides == {}


def test_factory_reset_removes_override_source():
    ChebfunPref.setDefaults(tech='trigtech', maxLength=65537, custom=9)
    ChebfunPref.setDefaults('maxLength', 'factory')
    assert ChebfunPref().maxLength == 65536
    factory = ChebfunPref.getFactoryDefaults()
    assert factory.tech == 'chebtech2' and factory._tech_overrides == {}
    assert ChebfunPref().custom == 9
    ChebfunPref.setDefaults('factory')
    assert 'custom' not in ChebfunPref().techPrefs
    with pytest.raises(KeyError):
        ChebfunPref.setDefaults('maxLength', 'factory')


def test_python_override_deletion_and_materialized_dict_semantics():
    p = ChebfunPref(tech='trigtech', maxLength=65537)
    del p.techPrefs['maxLength']
    assert p.maxLength == 65536
    with pytest.raises(KeyError):
        p.techPrefs.pop('maxLength')
    assert p.techPrefs.pop('maxLength', 10) == 10
    p.techPrefs.update(a=1, b=2)
    assert p.techPrefs.popitem() == ('b', 2)
    p.techPrefs.clear()
    assert p.maxLength == 65536 and p._tech_overrides == {}
    with pytest.raises(KeyError):
        p.techPrefs.popitem()
    detached = dict(p.techPrefs)
    del detached['maxLength']
    assert 'maxLength' not in detached


def test_whole_view_replacement_and_materialized_assignment():
    p = ChebfunPref(maxLength=42)
    p.techPrefs = {}
    assert p.maxLength == 65537
    p.techPrefs = dict(p.techPrefs)
    p.tech = 'trigtech'
    assert p.maxLength == 65537
    assert not any(k.startswith('_') for k in p.techPrefs)


def test_cheboppref_independent_session_and_factory():
    ChebfunPref.setDefaults(maxLength=42)
    ChebopPref.setDefaults(vectorize=False, maxLength=77)
    assert ChebfunPref().maxLength == 42
    assert ChebopPref().maxLength == 77 and ChebopPref().vectorize is False
    factory = ChebopPref.getFactoryDefaults()
    assert factory.maxLength == 65537 and factory.vectorize is True
    assert ChebopPref().maxLength == 77


def test_supported_class_tech_forms():
    from chebfunjax.tech.chebtech import Chebtech1, Chebtech2
    from chebfunjax.tech.trigtech import Trigtech

    for tech, expected in ((Chebtech1, 65537), (Chebtech2, 65537), (Trigtech, 65536)):
        assert ChebfunPref(tech=tech).maxLength == expected


def test_custom_explicit_provider_and_unknown_rejection():
    class Provider:
        @staticmethod
        def techPref():
            return {'maxLength': 29, 'custom': 3}

    p = ChebfunPref(tech=Provider, custom=4)
    assert p.maxLength == 29 and p.custom == 4
    with pytest.raises(ValueError, match='Unsupported preference technology'):
        ChebfunPref(tech='unknown').maxLength


def test_value_equality_does_not_determine_raw_provenance():
    p = ChebfunPref()
    q = ChebfunPref(maxLength=65537)
    assert p == q
    p.tech = q.tech = 'trigtech'
    assert p.maxLength == 65536 and q.maxLength == 65537
    assert isinstance(p.techPrefs, DotDict)


def test_stateful_provider_resolved_on_every_owner_read():
    state = {'maximum': 29, 'calls': 0}

    class Provider:
        @staticmethod
        def techPref():
            state['calls'] += 1
            return {'maxLength': state['maximum']}

    p = ChebfunPref(tech=Provider)
    assert p.maxLength == 29
    state['maximum'] = 31
    assert p.maxLength == 31
    state['maximum'] = 37
    assert dict(p.techPrefs) == {'maxLength': 37}
    assert p._tech_overrides == {}
    view = p.techPrefs
    before = state['calls']
    assert list(view.items()) == [('maxLength', 37)]
    assert list(view.items()) == [('maxLength', 37)]
    assert state['calls'] == before + 1


def test_raw_setters_do_not_resolve_unsupported_provider():
    p = ChebfunPref(tech='unknown')
    p.maxLength = 5
    p.techPrefs.sampleTest = False
    p.techPrefs['minSamples'] = 19
    p.techPrefs.update(custom=3)
    assert p._tech_overrides == {
        'maxLength': 5, 'sampleTest': False, 'minSamples': 19, 'custom': 3}
    p.tech = 'trigtech'
    assert p.maxLength == 5
    view = p.techPrefs
    p.tech = 'unknown'
    view['maxLength'] = 7
    with pytest.raises(ValueError, match='Unsupported preference technology'):
        p.maxLength


@pytest.mark.parametrize('tech,extra,maximum', [
    ('chebtech2', {'useTurbo': False}, 65537),
    ('trigtech', {'gridType': 2}, 65536),
])
def test_native_slots16_17_complete_selected_factory_fields(tech, extra, maximum):
    # Native tests/chebpref/test_chebfunpref.m114,117 compare complete techPref.
    # Expected executable fields independently transcribed from pinned techPref.
    expected = {
        'chebfuneps': 2**-52, 'minSamples': 17, 'maxLength': maximum,
        'fixedLength': None, 'extrapolate': False, 'sampleTest': True,
        'refinementFunction': 'nested', 'happinessCheck': 'standard', **extra}
    assert dict(ChebfunPref(tech=tech).techPrefs) == expected


@pytest.mark.parametrize('operation', ['or', 'ror', 'reversed'])
def test_python_union_and_reverse_resolve_dirty_view(operation):
    p = ChebfunPref()
    view = p.techPrefs
    p.tech = 'trigtech'
    if operation == 'or':
        result = view | {'custom': 3}
        assert result['maxLength'] == 65536 and result['custom'] == 3
        assert type(result) is dict
    elif operation == 'ror':
        result = {'maxLength': 7, 'custom': 3} | view
        assert result['maxLength'] == 65536 and result['custom'] == 3
        assert type(result) is dict
    else:
        assert list(reversed(view)) == list(reversed(dict(p.techPrefs)))
        assert 'gridType' in view and 'useTurbo' not in view
    assert p._tech_overrides == {}
