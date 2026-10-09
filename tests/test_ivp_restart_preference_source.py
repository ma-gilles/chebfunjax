"""Source factory field and persistence for the IVP restart preference.

Provenance: @cheboppref/cheboppref.m, factoryPrefs.ivpRestartSolver=true;
Chebfun7574c77. Python storage checks distinguish top fields from tech overrides.
"""
import pytest

from chebfunjax.chebpref import ChebopPref


@pytest.fixture(autouse=True)
def factory_session(monkeypatch):
    monkeypatch.setattr(ChebopPref, '_defaults', None)


def test_factory_restart_is_top_level():
    pref = ChebopPref.getFactoryDefaults()
    assert pref.ivpRestartSolver is True
    assert pref._top['ivpRestartSolver'] is True
    assert 'ivpRestartSolver' not in pref.techPrefs


@pytest.mark.parametrize('keyword', [False, True])
def test_restart_override_and_copy(keyword):
    pref = (ChebopPref(ivpRestartSolver=False) if keyword else
            ChebopPref({'ivpRestartSolver': False}))
    assert pref.ivpRestartSolver is False
    assert pref._top['ivpRestartSolver'] is False
    assert 'ivpRestartSolver' not in pref.techPrefs
    copied = ChebopPref(pref)
    copied.ivpRestartSolver = True
    assert pref.ivpRestartSolver is False
    assert copied.ivpRestartSolver is True


def test_restart_session_and_field_factory_reset():
    ChebopPref.setDefaults('ivpRestartSolver', False)
    assert ChebopPref().ivpRestartSolver is False
    ChebopPref.setDefaults('ivpRestartSolver', 'factory')
    assert ChebopPref().ivpRestartSolver is True
    assert 'ivpRestartSolver' not in ChebopPref().techPrefs
