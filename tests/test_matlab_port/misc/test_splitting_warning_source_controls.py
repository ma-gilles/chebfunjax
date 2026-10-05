"""Source warning/invalid-option behavior of MATLAB splitting.m.

Chebfun commit:7574c77680d7e82b79626300bf255498271a72df.
Existing tests/test_matlab_port/misc/test_splitting_matlab.py covers allsix
state/constructor assertions; these controls cover the omitted native warning
and warning-before-error behavior. Python FutureWarning is the visible adapter.
"""
from __future__ import annotations

import warnings

import pytest

import chebfunjax.chebpref as chebpref
from chebfunjax.chebpref import ChebfunPref


def _reset_warning_once(monkeypatch, value=False):
    monkeypatch.setattr(chebpref, '_SPLITTING_WARNING_EMITTED', value, raising=False)

def test_query_is_silent_and_setter_warns_once(monkeypatch):
    saved = ChebfunPref()
    _reset_warning_once(monkeypatch)
    try:
        ChebfunPref.setDefaults('factory')
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            assert chebpref.splitting() == 'off'
            assert seen == []
            assert chebpref.splitting('on') == 'off'
            assert chebpref.splitting('off') == 'on'
        assert len(seen) == 1
        assert seen[0].category is FutureWarning
        assert str(seen[0].message) == "The syntax 'splitting on' is deprecated.\nPlease see CHEBFUNPREF documentation for further details."
    finally:
        ChebfunPref.setDefaults(saved)

def test_invalid_option_warns_before_source_error_and_does_not_mutate(monkeypatch):
    saved = ChebfunPref()
    _reset_warning_once(monkeypatch)
    try:
        ChebfunPref.setDefaults('factory')
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            with pytest.raises(ValueError, match='CHEBFUN:splitting:UnknownOption: Unknown splitting option: only ON and OFF are valid options\\.'):
                chebpref.splitting('sometimes')
        assert len(seen) == 1
        assert ChebfunPref().splitting is False
    finally:
        ChebfunPref.setDefaults(saved)

def test_warning_promoted_to_error_does_not_consume_once_flag(monkeypatch):
    saved = ChebfunPref()
    _reset_warning_once(monkeypatch)
    try:
        ChebfunPref.setDefaults('factory')
        with warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            with pytest.raises(FutureWarning, match='deprecated'):
                chebpref.splitting('on')
        assert not chebpref._SPLITTING_WARNING_EMITTED
        assert ChebfunPref().splitting is False
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter('always')
            assert chebpref.splitting('on') == 'off'
        assert len(seen) == 1
        assert chebpref._SPLITTING_WARNING_EMITTED
        assert ChebfunPref().splitting is True
    finally:
        ChebfunPref.setDefaults(saved)
