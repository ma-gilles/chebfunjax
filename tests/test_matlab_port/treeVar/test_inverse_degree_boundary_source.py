"""Current TreeVar inverse-degree kernels against existing R2025b raw36 capture.

These supplemental captured cases complement the unchanged
49 original TreeVar predicates. The packet does not include acotd.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

DATA = json.loads((Path(__file__).parent / "data" /
                   "inverse_degree_native36.json").read_text())
NAMES = tuple(DATA["outputs_by_method"])
RECORDS = [
    {"method": method, "input": DATA["inputs"][i], "expected": expected}
    for method in NAMES
    for i, expected in enumerate(DATA["outputs_by_method"][method])
]


def _decode(word):
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _input(row):
    raw = row["input"]
    real, imag = _decode(raw["real_hex"]), _decode(raw["imag_hex"])
    return jnp.asarray(real if raw["isreal"] else complex(real, imag))


def _check_component(value, target):
    assert np.isnan(value) == np.isnan(target)
    assert np.isposinf(value) == np.isposinf(target)
    assert np.isneginf(value) == np.isneginf(target)
    if np.isfinite(target):
        assert abs(value-target) <= 8*np.finfo(float).eps*max(1., abs(target))
    if target == 0:
        assert value == 0
        assert np.signbit(value) == np.signbit(target)


@pytest.mark.parametrize("row", RECORDS,
                         ids=[r["method"]+"-"+r["input"]["label"]
                              for r in RECORDS])
def test_treevar_inverse_degree_native36_baseline(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    _check_component(float(np.real(got)), _decode(expected["real_hex"]))
    _check_component(float(np.imag(got)), _decode(expected["imag_hex"]))


def test_acotd_is_not_in_original_49_operation_packet():
    assert "acotd" not in DATA["names"]
