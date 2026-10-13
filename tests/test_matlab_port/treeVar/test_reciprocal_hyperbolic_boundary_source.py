"""Native reciprocal hyperbolic branch tests from the retained R2025b boundary packet.

Provenance
----------
MATLAB source: tests/treeVar/test_univariate.m, numeric built-ins dispatched
by @treeVar/treeVar.m; Chebfun 7574c77680d7e82b79626300bf255498271a72df.
Native packet: unary_boundary_native/capture.json (R2025b Update 2).
The finite-component comparison reuses the established 8eps*max(1,abs(x))
guard from neighboring TreeVar branch tests; original source bounds are not
rewritten by these supplemental primitive boundary cases.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from chebfunjax.operators.treevar import _UNARY_FNS

DATA = json.loads((Path(__file__).with_name("data")
                   / "reciprocal_hyperbolic_native.json").read_text(encoding="utf-8"))
RECORDS = DATA["records"]


def _decode(word: str) -> float:
    return struct.unpack(">d", bytes.fromhex(word))[0]


def _input(row):
    x = row["input"]
    re, im = _decode(x["real_hex"]), _decode(x["imag_hex"])
    return jnp.asarray(re if x["isreal"] else complex(re, im))


@pytest.mark.parametrize("row", RECORDS,
                         ids=[r["method"] + "-" + r["input"]["label"]
                              for r in RECORDS])
def test_native_reciprocal_hyperbolic_primitive(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    for part in ("real", "imag"):
        target = _decode(expected[part + "_hex"])
        value = float(getattr(got, part))
        assert np.isnan(value) == np.isnan(target)
        assert np.isposinf(value) == np.isposinf(target)
        assert np.isneginf(value) == np.isneginf(target)
        if np.isfinite(target):
            assert abs(value - target) <= (
                8 * np.finfo(float).eps * max(1.0, abs(target)))
        if target == 0:
            assert value == 0
            assert np.signbit(value) == np.signbit(target)


@pytest.mark.parametrize("method", ["acsch", "asech", "acoth"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_reciprocal_hyperbolic_float32_storage(method, complex_input):
    # Python dtype/shape adapter control, not a native single-precision oracle.
    value = jnp.asarray([.4+.3j, .7+.2j], dtype=jnp.complex64) if complex_input else jnp.asarray([.4, .7], dtype=jnp.float32)
    actual = _UNARY_FNS[method](value)
    expected_dtype = jnp.complex64 if complex_input or method == "acoth" else jnp.float32
    assert actual.shape == (2,)
    assert actual.dtype == expected_dtype
    assert bool(jnp.all(jnp.isfinite(actual)))
