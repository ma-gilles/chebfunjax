"""Raw-grid regression for MATLAB AAALawsonInit sampling arithmetic.

Provenance: MATLAB minimax.m AAALawsonInit, Chebfun commit 7574c77;
MATLAB R2025b linspace.m and positive round semantics.

The native fixture fixes the three successive input reference vectors. Only
AAA/SVD and extrema finding are mocked; this test does not fit a rational
function or qualify the complete initialization algorithm.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "references"
FIXTURE_NPZ = FIXTURE_DIR / "minimax_aaa_native_grid.npz"
FIXTURE_JSON = FIXTURE_DIR / "minimax_aaa_native_grid.json"


def _fixture():
    meta = json.loads(FIXTURE_JSON.read_text())
    assert meta["reference_commit"] == "7574c77680d7e82b79626300bf255498271a72df"
    digest = hashlib.sha256(FIXTURE_NPZ.read_bytes()).hexdigest()
    assert digest == meta["fixture_npz_sha256"]
    with np.load(FIXTURE_NPZ, allow_pickle=False) as data:
        return meta, {key: np.asarray(data[key]) for key in data.files}


def _raw_bits(values):
    return np.asarray(values, dtype=np.float64).reshape(-1).view(np.uint64)


def _mock_steps(monkeypatch, references):
    module = importlib.import_module("chebfunjax.utils.minimax")
    generated = []
    remaining = [np.asarray(ref, dtype=np.float64).copy() for ref in references]

    def fake_aaamn(_f, Z, _m, _n):
        generated.append(np.asarray(Z, dtype=np.float64).copy())
        return (lambda _x: np.asarray(_x, dtype=np.float64) * 0.0), np.zeros(1, dtype=np.float64)

    def fake_find_reference(_f, _rh, _m, _n, _z, _a, _b):
        assert remaining, "unexpected extra reference-finder call"
        return remaining.pop(0)

    monkeypatch.setattr(module, "_aaamn_lawson", fake_aaamn)
    monkeypatch.setattr(module, "_find_reference_rat", fake_find_reference)
    return module, generated, remaining


def _unused_target(_x):
    raise AssertionError("mocked grid test unexpectedly evaluated the target callback")


def test_aaa_lawson_grids_match_native_raw_bits_for_captured_references(monkeypatch):
    _, fixture = _fixture()
    refs = [fixture[f"xk{s}"] for s in (1, 2, 3)]
    module, generated, remaining = _mock_steps(monkeypatch, refs)
    final = module._aaa_lawson_init(
        _unused_target, 20, 20, -1.0, 1.0, refinement_source=_unused_target
    )

    assert not remaining
    assert len(generated) == 3
    for sweep in (1, 2, 3):
        native = fixture[f"Z{sweep}"]
        observed = np.asarray(generated[sweep - 1], dtype=np.float64).reshape(-1)
        assert observed.size == native.size
        np.testing.assert_array_equal(_raw_bits(observed), _raw_bits(native))
    np.testing.assert_array_equal(_raw_bits(final), _raw_bits(fixture["xk3"]))


def test_aaa_lawson_initial_count_uses_half_away_rounding(monkeypatch):
    refs = [np.linspace(-1.0, 1.0, 130) for _ in range(3)]
    module, generated, remaining = _mock_steps(monkeypatch, refs)
    module._aaa_lawson_init(_unused_target, 64, 64, -1.0, 1.0, refinement_source=_unused_target)
    assert not remaining
    assert len(generated) == 3
    assert np.asarray(generated[0]).size == 1563  # round(100000/64) is 1562.5.


def test_aaa_lawson_refinement_count_uses_half_away_rounding(monkeypatch):
    refs = [np.linspace(-1.0, 1.0, 14) for _ in range(3)]
    module, generated, remaining = _mock_steps(monkeypatch, refs)
    module._aaa_lawson_init(_unused_target, 6, 6, -1.0, 1.0, refinement_source=_unused_target)
    assert not remaining
    assert len(generated) == 3
    assert np.asarray(generated[0]).size == 16667
    # 16667/14 is 1190.5: MATLAB round gives num=1191; 13 joined
    # intervals then contain 13*(1191-1)+1 unique points.
    assert np.asarray(generated[1]).size == 15471
    assert np.asarray(generated[2]).size == 15471
