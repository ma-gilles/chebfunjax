"""Public Chebfun3 Phase 1 uses the no-argument technology default.

Provenance
----------
MATLAB source: @chebfun3/chebfun3f.m, preference initialization (lines 36-50)
Chebfun commit: 7574c77
"""

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
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


def test_session_min_samples_preference_does_not_change_phase1(monkeypatch):
    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults(ChebfunPref(minSamples=25))
        assert _first_aca_shape(monkeypatch) == (17, 36)
    finally:
        ChebfunPref.setDefaults(saved)


@pytest.mark.parametrize("tech", ["chebtech2", Chebtech1])
@pytest.mark.parametrize("min_samples", [5, 7, 25, 33])
def test_local_min_samples_preference_does_not_change_phase1(
    monkeypatch, tech, min_samples,
):
    pref = ChebfunPref(tech=tech, minSamples=min_samples)
    assert _first_aca_shape(monkeypatch, pref=pref) == (17, 36)


def test_explicit_keyword_overrides_preference_min_samples(monkeypatch):
    pref = ChebfunPref(minSamples=33)
    assert _first_aca_shape(monkeypatch, pref=pref, min_samples=41) == (41, 36)


@pytest.mark.parametrize("tech", ["chebtech2", Chebtech1])
def test_python_keyword_allows_small_initial_grid(monkeypatch, tech):
    # The Python keyword extends native grid selection; the subsequent
    # source algorithm has no artificial nine-point floor.
    pref = ChebfunPref(tech=tech)
    assert _first_aca_shape(monkeypatch, pref=pref, min_samples=7) == (7, 36)


@pytest.mark.parametrize("tech", ["chebtech2", Chebtech1])
def test_python_keyword_below_initial_rank_rejects_zero_random_bound(tech):
    # Explicit Python grid extension still obeys the source randi bound.
    pref = ChebfunPref(tech=tech)
    with pytest.raises(ValueError, match="positive integer sampling bound"):
        chebfun3(lambda x, y, z: x + y + z, pref=pref, min_samples=5)


_NATIVE_CAPTURE = json.loads((Path(__file__).parents[1] / "fixtures" /
    "chebfun3_min_samples_native_20261015.json").read_text())


@pytest.mark.parametrize("case", _NATIVE_CAPTURE["cases"],
                         ids=lambda case: case["caseName"])
def test_actual_tensor_sampling_matches_native_preference_capture(case):
    # Independent R2025b capture of pinned Chebfun; initial two 2x2x2 calls
    # are vectorization probes, followed by the actual rank-zero ACA tensor.
    assert not case["constructionError"]
    expected = case["events"][-1]
    seen = []

    def op(x, y, z):
        if x.ndim == 3 and x.shape != (2, 2, 2):
            seen.append((tuple(x.shape), tuple(y.shape), tuple(z.shape),
                         [(float(jnp.min(v)), float(jnp.max(v)))
                          for v in (x, y, z)]))
        return jnp.zeros_like(x + y + z)

    saved = ChebfunPref()
    try:
        ChebfunPref.setDefaults("factory")
        if case["caseName"] == "session25":
            ChebfunPref.setDefaults(ChebfunPref(minSamples=25))
        pref = ChebfunPref()
        pref.cheb3Prefs.sampleTest = False
        if case["caseName"].startswith("local"):
            pref.techPrefs.minSamples = case["configuredMinSamples"]
        result = chebfun3(op, pref=pref)
        assert float(result(.1, .2, .3)) == 0
    finally:
        ChebfunPref.setDefaults(saved)
    assert len(seen) == 1
    for got, axis in zip(seen[0][:3], "XYZ"):
        assert got == tuple(expected["shape" + axis])
    extrema = [(expected["min" + a], expected["max" + a]) for a in "XYZ"]
    np.testing.assert_allclose(seen[0][3], extrema, rtol=0, atol=2e-15)
