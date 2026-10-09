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
