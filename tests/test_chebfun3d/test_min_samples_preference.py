"""Public Chebfun3 Phase 1 grid size follows selected technology preferences.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m, preference initialization (lines 36-50)
Chebfun commit: 7574c77
"""

import pytest

from chebfunjax.chebfun3d import _chebfun3f_source as source3f
from chebfunjax.chebfun3d.chebfun3 import chebfun3
from chebfunjax.chebpref import ChebfunPref
from chebfunjax.tech.chebtech import Chebtech1


class _FirstTensorCaptured(Exception):
    pass


def _first_aca_shape(monkeypatch, **kwargs):
    shapes = []

    def capture(matrix, tolerance, max_iterations):
        shapes.append(tuple(matrix.shape))
        raise _FirstTensorCaptured

    monkeypatch.setattr(source3f, "aca", capture)
    with pytest.raises(_FirstTensorCaptured):
        chebfun3(lambda x, y, z: x + y + z, **kwargs)
    assert len(shapes) == 1
    return shapes[0]


def test_factory_min_samples_default_is_17(monkeypatch):
    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults("factory")
        assert _first_aca_shape(monkeypatch) == (17, 36)
    finally:
        ChebfunPref.setDefaults(saved)


def test_session_min_samples_preference_is_used(monkeypatch):
    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults(ChebfunPref(minSamples=25))
        assert _first_aca_shape(monkeypatch) == (25, 36)
    finally:
        ChebfunPref.setDefaults(saved)


def test_explicit_preference_min_samples_is_used(monkeypatch):
    pref = ChebfunPref(minSamples=33)
    assert _first_aca_shape(monkeypatch, pref=pref) == (33, 36)


def test_explicit_keyword_overrides_preference_min_samples(monkeypatch):
    pref = ChebfunPref(minSamples=33)
    assert _first_aca_shape(monkeypatch, pref=pref, min_samples=41) == (41, 36)


@pytest.mark.parametrize("tech", ["chebtech2", Chebtech1])
def test_source_polynomial_constructor_honors_small_min_samples(monkeypatch, tech):
    # MATLAB initializes n directly from tpref.minSamples; source-order
    # polynomial construction therefore has no artificial nine-point floor.
    pref = ChebfunPref(tech=tech, minSamples=7)
    assert _first_aca_shape(monkeypatch, pref=pref) == (7, 36)


@pytest.mark.parametrize("tech", ["chebtech2", Chebtech1])
def test_min_samples_below_initial_rank_preserves_native_error(tech):
    # Native r=[6,6,6], box=floor(5/6), then randi(0,1) errors.
    pref = ChebfunPref(tech=tech, minSamples=5)
    with pytest.raises(ValueError, match="positive integer sampling bound"):
        chebfun3(lambda x, y, z: x + y + z, pref=pref)
