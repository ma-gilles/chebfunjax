"""Native @chebfunpref/chebfunpref.m731 top-level FUNQUI default and writes."""
import pytest

from chebfunjax.chebpref import ChebfunPref


@pytest.fixture(autouse=True)
def preserve_preferences():
    saved = ChebfunPref()
    ChebfunPref.setDefaults('factory')
    yield
    ChebfunPref.setDefaults(saved)


def test_factory_funqui_false_is_top_level_and_not_a_tech_override():
    pref = ChebfunPref.getFactoryDefaults()
    assert pref.enableFunqui is False
    assert pref._top['enableFunqui'] is False
    assert 'enableFunqui' not in pref._tech_overrides
    assert 'enableFunqui' not in pref.techPrefs


def test_explicit_funqui_copy_preserves_raw_tech_omission():
    original = ChebfunPref({'enableFunqui': True, 'maxLength': 65537})
    copied = ChebfunPref(original)
    copied.enableFunqui = False
    assert original.enableFunqui is True and copied.enableFunqui is False
    assert dict(original._tech_overrides) == {'maxLength': 65537}
    assert dict(copied._tech_overrides) == {'maxLength': 65537}
    assert copied._top['enableFunqui'] is False


def test_session_funqui_inherits_and_explicit_override_is_private():
    ChebfunPref.setDefaults('enableFunqui', True)
    inherited = ChebfunPref()
    override = ChebfunPref({'enableFunqui': False})
    assert inherited.enableFunqui is True and override.enableFunqui is False
    assert ChebfunPref().enableFunqui is True
    assert not inherited._tech_overrides and not override._tech_overrides


def test_funqui_factory_reset_removes_no_unrelated_raw_preferences():
    ChebfunPref.setDefaults('enableFunqui', True, 'maxLength', 65537)
    ChebfunPref.setDefaults('enableFunqui', 'factory')
    pref = ChebfunPref()
    assert pref.enableFunqui is False
    assert dict(pref._tech_overrides) == {'maxLength': 65537}
    assert pref._top['enableFunqui'] is False
