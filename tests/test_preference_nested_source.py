"""Source ordered/two-tier setDefaults controls, pin7574c77.

manageDefaultPrefs683–710 mutates persistent state pair by pair. Python tuple
is an adapter for native cellstr alongside list. No numerical routines.
"""
import pytest

from chebfunjax.chebpref import ChebfunPref, ChebopPref


@pytest.fixture(autouse=True)
def isolated_sessions():
    saved = ChebfunPref._defaults, ChebopPref._defaults
    ChebfunPref.setDefaults('factory')
    ChebopPref.setDefaults('factory')
    yield
    ChebfunPref._defaults, ChebopPref._defaults = saved


@pytest.mark.parametrize('path_type', [list, tuple])
def test_two_tier_value_factory_and_sibling(path_type):
    path = path_type(('cheb2Prefs', 'maxRank'))
    ChebfunPref.setDefaults(path, 5)
    assert ChebfunPref().cheb2Prefs.maxRank == 5
    assert ChebfunPref().cheb2Prefs.sampleTest is True
    ChebfunPref.setDefaults(path, 'factory')
    assert ChebfunPref().cheb2Prefs.maxRank == 513


def test_ordered_duplicate_pairs_and_two_tier_pairs():
    ChebfunPref.setDefaults('chebfuneps', 1e-6, 'chebfuneps', 'factory',
                           ['cheb2Prefs', 'maxRank'], 5,
                           ['cheb2Prefs', 'maxRank'], 'factory')
    assert ChebfunPref().chebfuneps == 2**-52
    assert ChebfunPref().cheb2Prefs.maxRank == 513
    assert ChebfunPref()._tech_overrides == {}


def test_raw_tech_child_not_resolved_default():
    with pytest.raises(KeyError):
        ChebfunPref.setDefaults(['techPrefs', 'maxLength'], 8)
    ChebfunPref.setDefaults('maxLength', 9)
    ChebfunPref.setDefaults(['techPrefs', 'maxLength'], 8)
    assert ChebfunPref().maxLength == 8
    with pytest.raises(KeyError):
        ChebfunPref.setDefaults(['techPrefs', 'maxLength'], 'factory')
    assert ChebfunPref().maxLength == 8


def test_custom_existing_child_and_unavailable_factory():
    ChebfunPref.setDefaults({'cheb2Prefs': {'custom': 2}})
    ChebfunPref.setDefaults(['cheb2Prefs', 'custom'], 3)
    assert ChebfunPref().cheb2Prefs.custom == 3
    with pytest.raises(KeyError):
        ChebfunPref.setDefaults(['cheb2Prefs', 'custom'], 'factory')


class _NestedProvider:
    @staticmethod
    def techPref():
        return {'subPrefs': {'testSubPref': 'factory',
                             'child': {'leaf': 7}}}


def test_nested_provider_defaults_and_path_assignment_are_source_shaped():
    """Native techPref structs may expose nested provider preferences."""
    import copy

    pref = ChebfunPref(tech=_NestedProvider)
    assert pref.techPrefs.subPrefs.testSubPref == 'factory'
    assert pref.techPrefs.subPrefs.child.leaf == 7
    assert pref.techPrefs.subPrefs.child == {'leaf': 7}

    # @chebfunpref/subsasgn stores only the explicit nested path; a copied
    # preference remains a value copy and resolves its own provider defaults.
    copied = copy.deepcopy(pref)
    retained = pref.techPrefs.subPrefs.child
    retained.leaf = 11
    # Read from the held proxy without first refreshing through pref.techPrefs.
    assert retained.leaf == 11
    assert copied.techPrefs.subPrefs.child.leaf == 7

    # A materialized techPrefs copy is detached from the preference object.
    detached = pref.techPrefs.copy()
    detached['subPrefs']['child']['leaf'] = 13
    assert pref.techPrefs.subPrefs.child.leaf == 11

    # Technology switches retain explicit overrides, and switching back
    # resolves the custom provider without mutating its factory defaults.
    pref.tech = 'chebtech2'
    pref.tech = _NestedProvider
    assert pref.techPrefs.subPrefs.child.leaf == 11
    assert _NestedProvider.techPref()['subPrefs']['child']['leaf'] == 7


def test_explicit_nested_mapping_mutators_update_raw_preferences():
    pref = ChebfunPref({'techPrefs': {'subPrefs': {'first': 1, 'second': 2}}})
    nested = pref.techPrefs.subPrefs

    assert nested.pop('first') == 1
    assert nested.get('first', 'missing') == 'missing'
    assert nested.setdefault('third', 3) == 3
    assert nested.popitem() == ('third', 3)
    nested.clear()
    assert len(nested) == 0
    assert pref.techPrefs.subPrefs == {}


def test_nested_mapping_removal_returns_detached_value():
    pref = ChebfunPref({'techPrefs': {'subPrefs': {
        'first': {'leaf': 1}, 'last': {'child': {'leaf': 2}}}}})
    nested = pref.techPrefs.subPrefs

    popped = nested.pop('first')
    assert popped == {'leaf': 1}
    assert popped['leaf'] == 1
    key, item = nested.popitem()
    assert key == 'last'
    assert item == {'child': {'leaf': 2}}
    assert item['child']['leaf'] == 2
    assert nested == {}


def test_nested_provider_defaults_follow_session_factory_reset():
    ChebfunPref.setDefaults('tech', _NestedProvider)
    assert ChebfunPref().techPrefs.subPrefs.testSubPref == 'factory'
    ChebfunPref.setDefaults('factory')
    assert ChebfunPref().tech == 'chebtech2'
    assert 'subPrefs' not in ChebfunPref().techPrefs


def test_prior_pair_remains_committed_after_later_native_error():
    with pytest.raises(KeyError):
        ChebfunPref.setDefaults('maxLength', 19, ['cheb2Prefs', 'absent'], 2)
    assert ChebfunPref().maxLength == 19


def test_cheboppref_nested_session_independent():
    # Python inherited-field compatibility only; not native ChebopPref coverage.
    ChebopPref.setDefaults(['cheb2Prefs', 'maxRank'], 5)
    assert ChebopPref().cheb2Prefs.maxRank == 5
    assert ChebfunPref().cheb2Prefs.maxRank == 513
    ChebopPref.setDefaults(['cheb2Prefs', 'maxRank'], 'factory')
    assert ChebopPref().cheb2Prefs.maxRank == 513


@pytest.mark.parametrize('args', [(), ('domain',), (['cheb2Prefs'], 2)])
def test_native_argument_errors(args):
    with pytest.raises(TypeError):
        ChebfunPref.setDefaults(*args)
