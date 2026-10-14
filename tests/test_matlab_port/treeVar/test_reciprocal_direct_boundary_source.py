"""TreeVar direct reciprocal kernels against existing R2025b raw36 capture.

These supplemental comparisons retain the native packet component/storage
bounds. They do not replace the 49 original TreeVar predicates.
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
                   "reciprocal_direct_native.json").read_text())
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
def test_treevar_reciprocal_direct_native36(row):
    expected = row["expected"]
    assert expected["ok"]
    got = np.asarray(_UNARY_FNS[row["method"]](_input(row)))
    assert (not np.iscomplexobj(got)) == expected["isreal"]
    _check_component(float(np.real(got)), _decode(expected["real_hex"]))
    _check_component(float(np.imag(got)), _decode(expected["imag_hex"]))




@pytest.mark.parametrize("name", ["cot", "csc", "sec", "coth", "csch", "sech"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64, jnp.complex64, jnp.complex128])
def test_reciprocal_direct_precision_and_shape(name, dtype):
    """Supplemental finite values against an independent NumPy backend."""
    complex_input = jnp.issubdtype(dtype, jnp.complexfloating)
    raw = [[.4+.3j, -.7+.2j], [3j, -.25j]] if complex_input else [[.4, -.7], [2., -3.]]
    x = jnp.asarray(raw, dtype=dtype)
    result = _UNARY_FNS[name](x)
    assert result.shape == x.shape and result.dtype == dtype
    functions = {"cot": np.tan, "csc": np.sin, "sec": np.cos,
                 "coth": np.tanh, "csch": np.sinh, "sech": np.cosh}
    # Evaluate independent double-precision reference on exact rounded inputs.
    host = np.asarray(x).astype(complex if complex_input else float)
    expected = 1/functions[name](host)
    bound = 16*np.finfo(np.asarray(jnp.real(x)).dtype).eps
    assert np.max(np.abs(np.asarray(result)-expected)/np.maximum(1.,np.abs(expected))) < bound


@pytest.mark.parametrize("name", ["cot", "csc", "sec", "coth", "csch", "sech"])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_reciprocal_direct_signed_zero_poles(name, dtype):
    x = jnp.asarray([-0., 0.], dtype=dtype)
    result = np.asarray(_UNARY_FNS[name](x))
    assert result.dtype == np.asarray(x).dtype and result.shape == (2,)
    if name in ("sec", "sech"):
        np.testing.assert_array_equal(result, [1., 1.])
    else:
        assert np.isneginf(result[0]) and np.isposinf(result[1])
